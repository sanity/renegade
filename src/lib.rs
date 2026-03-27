mod diagnostics;
mod metric;
mod neighbor;
mod predict;
/// Vantage-point tree for metric-space nearest neighbor search.
pub mod vptree;

pub use diagnostics::{
    FeatureDiagnostics, ModelDiagnostics, NeighborDetail, OutputStats, PredictionDiagnostics,
};
pub use metric::LearnedMetric;
pub use neighbor::{Neighbor, Neighbors};
pub use predict::ExtrapolatedPrediction;

/// User implements this trait to define how distances are computed between data points.
///
/// Two methods must be implemented:
/// - `feature_distances`: per-feature distances in [0, 1] (for base KNN)
/// - `feature_values`: raw feature values (for metric learning)
///
/// **Important**: Both methods must describe the same features in the same order.
/// `feature_distances` returns pairwise distances while `feature_values` returns
/// raw values, but they must correspond to the same underlying features.
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
}

/// The core learner. Stores labeled training data and answers queries via KNN.
///
/// Designed for datasets up to ~100k points. Uses brute-force neighbor search
/// which is efficient up to this scale. For larger datasets, consider
/// data retention strategies (e.g., sliding window over recent events).
///
/// `query()` and `predict()` require `&mut self` because they trigger lazy
/// training (metric learning + K selection) on first call. Use `query_k()` and
/// `predict_k()` for immutable access with a manually specified K.
///
/// Training is amortized: the metric and K are only recomputed when the dataset
/// has doubled in size since the last computation. Call `force_retrain()` to
/// trigger recomputation manually.
pub struct Renegade<P: DataPoint> {
    // --- SoA layout for cache-friendly iteration ---
    /// Original data points (cold path — only accessed for feature_distances fallback).
    points: Vec<P>,
    /// Flat contiguous array of all feature values: [p0_f0, p0_f1, ..., p1_f0, p1_f1, ...].
    /// Length = num_entries * num_features. Indexed by `i * num_features + f`.
    values_flat: Vec<f64>,
    /// Output values, one per entry. Contiguous for cache-friendly access.
    outputs: Vec<f64>,
    /// Instance weights, one per entry. Default 1.0.
    instance_weights: Vec<f64>,
    /// Number of features per data point (0 until first point is added).
    num_features: usize,

    // --- Training state ---
    optimal_k: Option<usize>,
    learned_metric: Option<LearnedMetric>,
    /// Gaussian kernel bandwidth for regression. When set, predict() uses
    /// Gaussian-weighted mean over max_k neighbors instead of hard-k + 1/d.
    kernel_bandwidth: Option<f64>,
    /// VP-tree index for fast queries.
    vp_index: Option<vptree::VpTree>,
    /// Number of entries when optimal_k / metric were last computed.
    computed_at: usize,
    /// Number of entries when the VP-tree was last built.
    vp_built_at: usize,
}

/// Minimum number of data points before learning a metric.
const MIN_POINTS_FOR_METRIC: usize = 10;

/// Minimum entries to build a VP-tree (below this, brute force is fine).
const VP_TREE_THRESHOLD: usize = 3;

impl<P: DataPoint + Clone> Renegade<P> {
    /// Create a new empty learner.
    pub fn new() -> Self {
        Renegade {
            points: Vec::new(),
            values_flat: Vec::new(),
            outputs: Vec::new(),
            instance_weights: Vec::new(),
            num_features: 0,
            optimal_k: None,
            learned_metric: None,
            kernel_bandwidth: None,
            vp_index: None,
            computed_at: 0,
            vp_built_at: 0,
        }
    }

    /// Add a labeled data point with default weight 1.0.
    pub fn add(&mut self, point: P, output: f64) {
        self.add_weighted(point, output, 1.0);
    }

    /// Add a labeled data point with a specific instance weight.
    /// Higher weight means this point has more influence on predictions.
    /// Weight must be positive.
    pub fn add_weighted(&mut self, point: P, output: f64, weight: f64) {
        debug_assert!(weight > 0.0, "Instance weight must be positive");
        let values = point.feature_values();
        if self.num_features == 0 {
            self.num_features = values.len();
            debug_assert_eq!(
                values.len(),
                point.feature_distances(&point).len(),
                "feature_values() and feature_distances() must return the same number of features"
            );
        } else {
            debug_assert_eq!(
                values.len(),
                self.num_features,
                "All data points must have the same number of features"
            );
        }
        self.values_flat.extend_from_slice(&values);
        self.outputs.push(output);
        self.instance_weights.push(weight);
        self.points.push(point);

        // Invalidate metric/K if dataset has grown 50% since last training
        if self.computed_at > 0 && self.len() >= self.computed_at + self.computed_at / 2 {
            self.optimal_k = None;
            self.learned_metric = None;
            self.kernel_bandwidth = None;
            self.vp_index = None;
            self.vp_built_at = 0;
        }

        // Rebuild VP-tree (cheap) when unindexed tail exceeds 20% of indexed points
        if self.vp_built_at > 0 {
            let tail = self.len() - self.vp_built_at;
            if tail > self.vp_built_at / 5 {
                self.rebuild_vp_tree();
            }
        }
    }

    /// Number of training points.
    #[inline]
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether the learner has no training data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Remove entries that don't satisfy the predicate. Useful for expiring
    /// stale data (e.g., sliding window over recent events).
    /// Invalidates cached K and metric.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&P, f64) -> bool,
    {
        let n = self.len();
        let nf = self.num_features;
        let mut write = 0;
        for read in 0..n {
            if f(&self.points[read], self.outputs[read]) {
                if write != read {
                    self.points.swap(write, read);
                    self.outputs.swap(write, read);
                    self.instance_weights.swap(write, read);
                    self.values_flat
                        .copy_within(read * nf..(read + 1) * nf, write * nf);
                }
                write += 1;
            }
        }
        self.points.truncate(write);
        self.outputs.truncate(write);
        self.instance_weights.truncate(write);
        self.values_flat.truncate(write * nf);
        self.invalidate();
    }

    /// Force recomputation of the metric and K on the next query.
    pub fn force_retrain(&mut self) {
        self.invalidate();
    }

    /// Clear all cached training state.
    fn invalidate(&mut self) {
        self.optimal_k = None;
        self.learned_metric = None;
        self.kernel_bandwidth = None;
        self.vp_index = None;
        self.vp_built_at = 0;
    }

    /// Rebuild just the VP-tree (cheap) without retraining metric/K.
    fn rebuild_vp_tree(&mut self) {
        let n = self.len();
        if n >= VP_TREE_THRESHOLD {
            self.vp_index = Some(vptree::VpTree::build(n, &|a, b| {
                self.distance_between(a, b)
            }));
            self.vp_built_at = n;
        }
    }

    /// Get the cached feature values for entry i as a slice.
    #[inline]
    fn entry_values(&self, i: usize) -> &[f64] {
        let nf = self.num_features;
        &self.values_flat[i * nf..(i + 1) * nf]
    }

    /// Compute distance between a query (given as values slice) and entry i.
    #[inline]
    fn distance_to_entry(&self, query_values: &[f64], query: &P, i: usize) -> f64 {
        match &self.learned_metric {
            Some(metric) => metric.distance(query_values, self.entry_values(i)),
            None => {
                let feat_dists = query.feature_distances(&self.points[i]);
                if feat_dists.is_empty() {
                    return 0.0;
                }
                feat_dists.iter().sum::<f64>() / feat_dists.len() as f64
            }
        }
    }

    /// Compute distance between entries i and j.
    #[inline]
    fn distance_between(&self, i: usize, j: usize) -> f64 {
        match &self.learned_metric {
            Some(metric) => metric.distance(self.entry_values(i), self.entry_values(j)),
            None => {
                let feat_dists = self.points[i].feature_distances(&self.points[j]);
                if feat_dists.is_empty() {
                    return 0.0;
                }
                feat_dists.iter().sum::<f64>() / feat_dists.len() as f64
            }
        }
    }

    /// Find the k nearest neighbors to a query point.
    /// Returns neighbors sorted by distance (closest first).
    /// Uses VP-tree for indexed points, plus brute-force scan of any points
    /// added since the tree was built.
    pub fn query_k(&self, query: &P, k: usize) -> Neighbors {
        let query_values = query.feature_values();
        let n = self.len();

        let results = if let Some(ref vp) = self.vp_index {
            let query_dist = |i: usize| self.distance_to_entry(&query_values, query, i);

            // Search VP-tree for indexed points
            let mut results = vp.query_nearest(k, &query_dist);

            // Brute-force scan any points added after the tree was built
            if self.vp_built_at < n {
                for i in self.vp_built_at..n {
                    let dist = self.distance_to_entry(&query_values, query, i);
                    if results.len() < k {
                        results.push((i, dist));
                        results.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    } else if let Some(worst) = results.last() {
                        if dist < worst.1 {
                            results.pop();
                            results.push((i, dist));
                            results.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                    }
                }
            }

            results
        } else {
            // No VP-tree: brute force all points
            let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n);
            for i in 0..n {
                let dist = self.distance_to_entry(&query_values, query, i);
                distances.push((i, dist));
            }
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(k);
            distances
        };

        let neighbors = results
            .into_iter()
            .map(|(i, dist)| Neighbor {
                distance: dist,
                output: self.outputs[i],
                weight: self.instance_weights[i],
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

    /// Predict output using automatically determined K and weighted mean.
    /// For regression, may use Gaussian kernel weighting if it was selected
    /// during training as superior to hard-k + inverse-distance.
    pub fn predict(&mut self, query: &P) -> f64 {
        self.ensure_trained();
        let k = self.optimal_k.unwrap();
        if let Some(h) = self.kernel_bandwidth {
            // Gaussian kernel: query max_k neighbors so the kernel has a full
            // neighborhood to weight. The kernel itself does the "soft cutoff" —
            // distant neighbors contribute exponentially less.
            let max_k = (self.len() as f64).sqrt().ceil() as usize;
            let neighbors = self.query_k(query, max_k);
            neighbors.gaussian_weighted_mean(h)
        } else {
            let neighbors = self.query_k(query, k);
            neighbors.weighted_mean()
        }
    }

    /// Predict output using distance-trend extrapolation (auto K).
    pub fn predict_extrapolated(&mut self, query: &P) -> ExtrapolatedPrediction {
        let neighbors = self.query(query);
        neighbors.extrapolate()
    }

    /// Predict output for a query point using specified k and weighted mean.
    pub fn predict_k(&self, query: &P, k: usize) -> f64 {
        let neighbors = self.query_k(query, k);
        neighbors.weighted_mean()
    }

    /// Predict output for a query point using specified k and distance-trend extrapolation.
    pub fn predict_k_extrapolated(&self, query: &P, k: usize) -> ExtrapolatedPrediction {
        let neighbors = self.query_k(query, k);
        neighbors.extrapolate()
    }

    /// Ensure the metric and K are trained. Recomputes if needed.
    /// Learns the metric, then compares LOO error with and without it.
    /// Only keeps the metric if it actually improves predictions.
    /// For regression, also evaluates Gaussian kernel weighting and uses it
    /// if it outperforms hard-k + inverse-distance.
    fn ensure_trained(&mut self) {
        if self.optimal_k.is_some() {
            return;
        }

        if self.len() >= MIN_POINTS_FOR_METRIC {
            // Compute best K (and bandwidth for regression) without metric.
            self.learned_metric = None;
            let (k_no_metric, error_no_metric, bw_no_metric) =
                self.compute_optimal_k_and_bandwidth();

            // Learn metric and compute best K (and bandwidth) with it
            let candidate_metric = self.learn_metric();
            self.learned_metric = Some(candidate_metric);
            let (k_with_metric, error_with_metric, bw_with_metric) =
                self.compute_optimal_k_and_bandwidth();

            // Pick the globally best configuration across all 4 combinations:
            // {no-metric, metric} × {hard-k, gaussian}
            let best_no_metric = match bw_no_metric {
                Some((_, bw_err)) if bw_err < error_no_metric => bw_err,
                _ => error_no_metric,
            };
            let best_with_metric = match bw_with_metric {
                Some((_, bw_err)) if bw_err < error_with_metric => bw_err,
                _ => error_with_metric,
            };

            if best_with_metric < best_no_metric {
                // Keep metric
                self.optimal_k = Some(k_with_metric);
                if let Some((h, bw_err)) = bw_with_metric {
                    if bw_err < error_with_metric {
                        self.kernel_bandwidth = Some(h);
                    }
                }
            } else {
                // No metric
                self.learned_metric = None;
                self.optimal_k = Some(k_no_metric);
                if let Some((h, bw_err)) = bw_no_metric {
                    if bw_err < error_no_metric {
                        self.kernel_bandwidth = Some(h);
                    }
                }
            }
        } else {
            self.learned_metric = None;
            let k = self.compute_optimal_k();
            self.optimal_k = Some(k);
        }

        // Build VP-tree index for fast queries
        self.rebuild_vp_tree();

        self.computed_at = self.len();
    }

    /// Get the current optimal K, training if necessary.
    pub fn get_optimal_k(&mut self) -> usize {
        self.ensure_trained();
        self.optimal_k.unwrap()
    }

    /// Learn the metric from training data using effect-space isotonic regressions.
    fn learn_metric(&self) -> LearnedMetric {
        use metric::TrainingPoint;

        let points: Vec<TrainingPoint> = (0..self.len())
            .map(|i| TrainingPoint {
                features: self.entry_values(i).to_vec(),
                output: self.outputs[i],
            })
            .collect();

        LearnedMetric::learn(&points)
    }

    /// Compute optimal K via leave-one-out cross-validation.
    /// Computes distances once per eval point, then evaluates all K values
    /// from the sorted distance list.
    /// For regression, also sweeps Gaussian bandwidth candidates in the same
    /// pass (zero extra distance computations).
    /// Returns (best_k, Option<(bandwidth, bandwidth_error)>).
    fn compute_optimal_k(&self) -> usize {
        self.compute_optimal_k_and_bandwidth().0
    }

    /// Joint optimization of k and bandwidth. Returns:
    /// (best_k, best_k_mse, Option<(best_bandwidth, best_bandwidth_mse)>)
    fn compute_optimal_k_and_bandwidth(&self) -> (usize, f64, Option<(f64, f64)>) {
        let n = self.len();
        if n <= 2 {
            return (n.max(1), f64::MAX, None);
        }

        let max_k = (n as f64).sqrt().ceil() as usize;
        let max_k = max_k.max(1).min(n - 1);

        let is_classification = self.detect_classification();

        let max_eval = 200.min(n);
        let step = if n > max_eval { n / max_eval } else { 1 };

        // Collect sorted distances for each eval point (shared by k and bandwidth sweeps)
        let eval_data: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
            .step_by(step)
            .take(max_eval)
            .map(|i| {
                let mut distances: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, self.distance_between(i, j)))
                    .collect();
                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                distances.truncate(max_k);
                (i, distances)
            })
            .collect();

        let count = eval_data.len();
        if count == 0 {
            return (1, f64::MAX, None);
        }

        // Sweep k values
        let mut errors_by_k = vec![0.0f64; max_k + 1];

        for &(i, ref distances) in &eval_data {
            if is_classification {
                // Weighted class voting — matches class_votes() behavior
                let mut votes: Vec<(f64, f64)> = Vec::new(); // (class, total_weight)
                for k in 1..=max_k.min(distances.len()) {
                    let (j, dist) = distances[k - 1];
                    let val = self.outputs[j];
                    let w = if dist == 0.0 {
                        self.instance_weights[j] * 1e6
                    } else {
                        self.instance_weights[j] / dist
                    };
                    if let Some(entry) = votes.iter_mut().find(|(v, _)| (*v - val).abs() < 1e-10) {
                        entry.1 += w;
                    } else {
                        votes.push((val, w));
                    }
                    let predicted = votes
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
                    if (predicted - self.outputs[i]).abs() > 0.5 {
                        errors_by_k[k] += 1.0;
                    }
                }
            } else {
                // Inverse-distance weighting with instance weights — matches weighted_mean()
                let mut weight_sum = 0.0;
                let mut value_sum = 0.0;
                let mut exact_w = 0.0;
                let mut exact_v = 0.0;
                let mut has_exact = false;

                for k in 1..=max_k.min(distances.len()) {
                    let (j, dist) = distances[k - 1];

                    if dist == 0.0 {
                        has_exact = true;
                        exact_w += self.instance_weights[j];
                        exact_v += self.instance_weights[j] * self.outputs[j];
                    } else if !has_exact {
                        let w = self.instance_weights[j] / dist;
                        weight_sum += w;
                        value_sum += w * self.outputs[j];
                    }

                    let predicted = if has_exact {
                        if exact_w > 0.0 {
                            exact_v / exact_w
                        } else {
                            self.outputs[j]
                        }
                    } else if weight_sum > 0.0 {
                        value_sum / weight_sum
                    } else {
                        continue;
                    };

                    let err = predicted - self.outputs[i];
                    errors_by_k[k] += err * err;
                }
            }
        }

        let mut best_k = 1;
        let mut best_k_error = f64::MAX;
        for (k, &err) in errors_by_k.iter().enumerate().skip(1) {
            let error = err / count as f64;
            if error < best_k_error {
                best_k_error = error;
                best_k = k;
            }
        }

        // For regression, also sweep Gaussian bandwidth candidates (no extra distance computation)
        let bandwidth_result = if !is_classification {
            // Build bandwidth candidates from distance percentiles
            let mut all_dists: Vec<f64> = Vec::new();
            for (_, distances) in &eval_data {
                for &(_, d) in distances {
                    if d > 0.0 {
                        all_dists.push(d);
                    }
                }
            }

            if all_dists.is_empty() {
                None
            } else {
                all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let h_candidates: Vec<f64> = (1..=20)
                    .map(|t| {
                        let pct = t as f64 / 21.0;
                        let idx = (pct * all_dists.len() as f64) as usize;
                        all_dists[idx.min(all_dists.len() - 1)]
                    })
                    .collect();

                let mut best_h = h_candidates[0];
                let mut best_h_error = f64::MAX;

                for &h in &h_candidates {
                    let h2 = 2.0 * h * h;
                    let mut total_error = 0.0;

                    for &(i, ref distances) in &eval_data {
                        let mut weight_sum = 0.0;
                        let mut value_sum = 0.0;
                        let mut exact_match = None;

                        for &(j, dist) in distances {
                            if dist == 0.0 {
                                exact_match = Some(self.outputs[j]);
                                break;
                            }
                            let w = (-dist * dist / h2).exp() * self.instance_weights[j];
                            if w < 1e-15 {
                                break;
                            }
                            weight_sum += w;
                            value_sum += w * self.outputs[j];
                        }

                        let predicted = if let Some(v) = exact_match {
                            v
                        } else if weight_sum > 0.0 {
                            value_sum / weight_sum
                        } else if let Some(&(j, _)) = distances.first() {
                            self.outputs[j]
                        } else {
                            continue;
                        };

                        let err = predicted - self.outputs[i];
                        total_error += err * err;
                    }

                    let avg_error = total_error / count as f64;
                    if avg_error < best_h_error {
                        best_h_error = avg_error;
                        best_h = h;
                    }
                }

                Some((best_h, best_h_error))
            }
        } else {
            None
        };

        (best_k, best_k_error, bandwidth_result)
    }

    /// Detect whether this is a classification or regression problem.
    /// Heuristic: all integer outputs, ≤20 distinct values, AND the ratio of
    /// distinct values to dataset size is low enough to look categorical.
    /// This avoids misfiring on integer-valued regression targets like
    /// ratings (1-5), counts, or ages.
    fn detect_classification(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        let all_integer = self.outputs.iter().all(|&o| (o - o.round()).abs() < 1e-6);

        if !all_integer {
            return false;
        }

        let mut distinct: Vec<f64> = Vec::new();
        for &o in &self.outputs {
            let val = o.round();
            if !distinct.iter().any(|&v| (v - val).abs() < 1e-10) {
                distinct.push(val);
                if distinct.len() > 20 {
                    return false;
                }
            }
        }

        let n = self.len();
        let n_distinct = distinct.len();

        // With very few data points, can't reliably distinguish — default to regression
        // unless there are clearly only 2-3 classes.
        if n < 10 {
            return n_distinct <= 3;
        }

        // For larger datasets: if distinct values are a large fraction of the data,
        // it's more likely integer regression (e.g., 50 distinct values out of 200 points).
        // Classification datasets typically have n_distinct << sqrt(n).
        let max_classes = (n as f64).sqrt().ceil() as usize;
        n_distinct <= max_classes.min(20)
    }
}

impl<P: DataPoint + Clone> Default for Renegade<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
