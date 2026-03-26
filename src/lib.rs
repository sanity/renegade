mod metric;
mod neighbor;
mod predict;

pub use metric::LearnedMetric;
pub use neighbor::{Neighbor, Neighbors};
pub use predict::ExtrapolatedPrediction;

/// User implements this trait to define how distances are computed between data points.
///
/// Two methods must be implemented:
/// - `feature_distances`: per-feature distances in [0, 1] (for base KNN)
/// - `feature_values`: raw feature values (for metric learning)
///
/// For numeric features, distances are typically |a - b| / (max - min).
/// For categorical features: 0.0 if same, 1.0 if different.
/// Custom distance functions (edit distance, Jaccard, etc.) are fine as long as
/// they're normalized to [0, 1].
pub trait DataPoint {
    /// Per-feature distances between this point and another, each in [0, 1].
    fn feature_distances(&self, other: &Self) -> Vec<f64>;

    /// Raw feature values for this point, used by the metric learner.
    /// Each feature should be a numeric value. For categorical features,
    /// use a numeric encoding (e.g., 0, 1, 2, ...).
    fn feature_values(&self) -> Vec<f64>;

    /// Number of features.
    fn num_features(&self) -> usize;
}

/// The core learner. Stores labeled training data and answers queries via KNN.
///
/// Designed for datasets up to ~100k points. Uses brute-force neighbor search
/// which is efficient up to this scale. For larger datasets, consider
/// data retention strategies (e.g., sliding window over recent events).
pub struct Renegade<P: DataPoint> {
    entries: Vec<Entry<P>>,
    optimal_k: Option<usize>,
    learned_metric: Option<LearnedMetric>,
    /// Number of entries when optimal_k / metric were last computed.
    computed_at: usize,
}

struct Entry<P: DataPoint> {
    point: P,
    output: f64,
}

/// Minimum number of data points before learning a metric.
const MIN_POINTS_FOR_METRIC: usize = 10;

impl<P: DataPoint + Clone> Renegade<P> {
    /// Create a new empty learner.
    pub fn new() -> Self {
        Renegade {
            entries: Vec::new(),
            optimal_k: None,
            learned_metric: None,
            computed_at: 0,
        }
    }

    /// Add a labeled data point. Invalidates cached K and metric if the dataset
    /// has grown significantly since last computation.
    pub fn add(&mut self, point: P, output: f64) {
        self.entries.push(Entry { point, output });
        // Invalidate if dataset has doubled since last computation
        if self.computed_at > 0 && self.entries.len() >= self.computed_at * 2 {
            self.optimal_k = None;
            self.learned_metric = None;
        }
    }

    /// Number of training points.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the learner has no training data.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute distance between two points using the learned metric if available,
    /// otherwise fall back to simple mean of per-feature distances.
    fn distance(&self, a: &P, b: &P) -> f64 {
        match &self.learned_metric {
            Some(metric) => metric.distance(&a.feature_values(), &b.feature_values()),
            None => {
                let feat_dists = a.feature_distances(b);
                feat_dists.iter().sum::<f64>() / feat_dists.len() as f64
            }
        }
    }

    /// Find the k nearest neighbors to a query point.
    /// Returns neighbors sorted by distance (closest first).
    pub fn query_k(&self, query: &P, k: usize) -> Neighbors {
        let mut distances: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (i, self.distance(query, &entry.point)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        let neighbors = distances
            .into_iter()
            .map(|(i, dist)| Neighbor {
                distance: dist,
                output: self.entries[i].output,
            })
            .collect();

        Neighbors { neighbors }
    }

    /// Find nearest neighbors using automatically determined K.
    /// Learns the metric and computes optimal K if needed.
    pub fn query(&mut self, query: &P) -> Neighbors {
        self.ensure_trained();
        let k = self.optimal_k.unwrap();
        self.query_k(query, k)
    }

    /// Predict output using automatically determined K and distance-trend extrapolation.
    pub fn predict(&mut self, query: &P) -> ExtrapolatedPrediction {
        let neighbors = self.query(query);
        neighbors.extrapolate()
    }

    /// Predict output for a query point using specified k and distance-trend extrapolation.
    pub fn predict_k(&self, query: &P, k: usize) -> ExtrapolatedPrediction {
        let neighbors = self.query_k(query, k);
        neighbors.extrapolate()
    }

    /// Ensure the metric and K are trained. Recomputes if needed.
    /// Learns the metric, then compares LOO error with and without it.
    /// Only keeps the metric if it actually improves predictions.
    fn ensure_trained(&mut self) {
        if self.optimal_k.is_some() {
            return;
        }

        if self.entries.len() >= MIN_POINTS_FOR_METRIC {
            // Compute best K without metric
            self.learned_metric = None;
            let k_no_metric = self.compute_optimal_k();
            let is_classification = self.detect_classification();
            let error_no_metric = self.loo_error(k_no_metric, is_classification);

            // Learn metric and compute best K with it
            let candidate_metric = self.learn_metric();
            self.learned_metric = Some(candidate_metric);
            let k_with_metric = self.compute_optimal_k();
            let error_with_metric = self.loo_error(k_with_metric, is_classification);

            // Only keep metric if it strictly improves LOO error
            if error_with_metric < error_no_metric {
                self.optimal_k = Some(k_with_metric);
            } else {
                self.learned_metric = None;
                self.optimal_k = Some(k_no_metric);
            }
        } else {
            self.learned_metric = None;
            let k = self.compute_optimal_k();
            self.optimal_k = Some(k);
        }

        self.computed_at = self.entries.len();
    }

    /// Get the current optimal K, training if necessary.
    pub fn get_optimal_k(&mut self) -> usize {
        self.ensure_trained();
        self.optimal_k.unwrap()
    }

    /// Learn the metric from training data using effect-space isotonic regressions.
    fn learn_metric(&self) -> LearnedMetric {
        use metric::TrainingPoint;

        let points: Vec<TrainingPoint> = self
            .entries
            .iter()
            .map(|e| TrainingPoint {
                features: e.point.feature_values(),
                output: e.output,
            })
            .collect();

        LearnedMetric::learn(&points)
    }

    /// Compute optimal K via leave-one-out cross-validation.
    fn compute_optimal_k(&self) -> usize {
        let n = self.entries.len();
        if n <= 2 {
            return n.max(1);
        }

        let max_k = (n as f64).sqrt().ceil() as usize;
        let max_k = max_k.max(1).min(n - 1);

        let mut best_k = 1;
        let mut best_error = f64::MAX;

        let is_classification = self.detect_classification();

        for k in 1..=max_k {
            let error = self.loo_error(k, is_classification);
            if error < best_error {
                best_error = error;
                best_k = k;
            }
        }

        best_k
    }

    /// Heuristic: if all output values are "small integers" (within epsilon of
    /// an integer), treat as classification. Otherwise regression.
    fn detect_classification(&self) -> bool {
        self.entries
            .iter()
            .all(|e| (e.output - e.output.round()).abs() < 1e-6)
    }

    /// Compute leave-one-out error for a given K.
    /// For large datasets, evaluates a deterministic subsample to keep training fast.
    fn loo_error(&self, k: usize, is_classification: bool) -> f64 {
        let n = self.entries.len();
        let mut total_error = 0.0;

        // Cap LOO evaluation to keep training fast.
        // Each eval point computes distance to all n points, so cost is max_eval * n.
        let max_eval = 200.min(n);
        let step = if n > max_eval { n / max_eval } else { 1 };
        let mut count = 0;

        for i in (0..n).step_by(step).take(max_eval) {
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = self.distance(&self.entries[i].point, &self.entries[j].point);
                    (j, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(k);

            if is_classification {
                let mut counts: Vec<(f64, usize)> = Vec::new();
                for &(j, _) in &distances {
                    let val = self.entries[j].output;
                    if let Some(entry) = counts.iter_mut().find(|(v, _)| (*v - val).abs() < 1e-10) {
                        entry.1 += 1;
                    } else {
                        counts.push((val, 1));
                    }
                }
                let predicted = counts.iter().max_by_key(|(_, count)| *count).unwrap().0;
                if (predicted - self.entries[i].output).abs() > 0.5 {
                    total_error += 1.0;
                }
            } else {
                let mut weight_sum = 0.0;
                let mut value_sum = 0.0;
                let mut exact_match = None;
                for &(j, dist) in &distances {
                    if dist == 0.0 {
                        exact_match = Some(self.entries[j].output);
                        break;
                    }
                    let w = 1.0 / dist;
                    weight_sum += w;
                    value_sum += w * self.entries[j].output;
                }
                let predicted = exact_match.unwrap_or(value_sum / weight_sum);
                let err = predicted - self.entries[i].output;
                total_error += err * err;
            }
            count += 1;
        }

        if count == 0 {
            return f64::MAX;
        }
        total_error / count as f64
    }
}

impl<P: DataPoint + Clone> Default for Renegade<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
