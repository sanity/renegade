use crate::metric::LearnedMetric;
use crate::{DataPoint, Renegade};

/// Snapshot of the model's current state for diagnostics/dashboards.
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    /// Number of training points.
    pub num_entries: usize,
    /// Current auto-selected K (None if not yet trained).
    pub optimal_k: Option<usize>,
    /// Whether the learned metric is active (vs baseline Gower distance).
    pub metric_active: bool,
    /// Gaussian kernel bandwidth (None = using hard-k + 1/d weighting).
    pub kernel_bandwidth: Option<f64>,
    /// Number of entries when the model was last trained.
    pub trained_at: usize,
    /// Number of entries added since last training.
    pub entries_since_training: usize,
    /// Whether the model detected a classification task (vs regression).
    pub is_classification: bool,
    /// Per-feature metric diagnostics (only available when metric is active).
    pub feature_metrics: Option<Vec<FeatureDiagnostics>>,
    /// Output value statistics.
    pub output_stats: OutputStats,
}

/// Diagnostics for a single feature's learned metric.
#[derive(Debug, Clone)]
pub struct FeatureDiagnostics {
    /// Feature index.
    pub index: usize,
    /// Weight assigned to this feature (0.0 = noise, higher = more predictive).
    pub weight: f64,
    /// The effect curve: (feature_value, predicted_output) points from isotonic regression.
    /// Sorted by feature_value.
    pub effect_curve: Vec<(f64, f64)>,
}

/// Statistics about the output values in the training set.
#[derive(Debug, Clone)]
pub struct OutputStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Number of distinct output values.
    pub num_distinct: usize,
}

/// Diagnostics for a single prediction query.
#[derive(Debug, Clone)]
pub struct PredictionDiagnostics {
    /// The predicted output value (weighted mean).
    pub prediction: f64,
    /// The K used for this prediction.
    pub k: usize,
    /// The nearest neighbors used, sorted by distance.
    pub neighbors: Vec<NeighborDetail>,
}

/// Detail about a single neighbor in a prediction.
#[derive(Debug, Clone)]
pub struct NeighborDetail {
    /// Distance from the query point.
    pub distance: f64,
    /// Output value of this neighbor.
    pub output: f64,
    /// Per-feature distances (only available when not using learned metric).
    pub feature_distances: Option<Vec<f64>>,
}

impl<P: DataPoint + Clone> Renegade<P> {
    /// Get a snapshot of the model's current state for diagnostics.
    pub fn diagnostics(&self) -> ModelDiagnostics {
        let output_stats = self.compute_output_stats();
        let is_classification = self.detect_classification();

        let feature_metrics = self
            .learned_metric
            .as_ref()
            .map(|metric| metric.feature_diagnostics());

        ModelDiagnostics {
            num_entries: self.len(),
            optimal_k: self.optimal_k,
            metric_active: self.learned_metric.is_some(),
            kernel_bandwidth: self.kernel_bandwidth,
            trained_at: self.computed_at,
            entries_since_training: self.len().saturating_sub(self.computed_at),
            is_classification,
            feature_metrics,
            output_stats,
        }
    }

    /// Get detailed diagnostics for a specific prediction.
    pub fn predict_with_diagnostics(&self, query: &P, k: usize) -> PredictionDiagnostics {
        let query_values = query.feature_values();
        let n = self.len();
        let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let dist = self.distance_to_entry(&query_values, query, i);
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        let neighbors: Vec<NeighborDetail> = distances
            .iter()
            .map(|&(i, dist)| {
                let feature_distances = if self.learned_metric.is_none() {
                    Some(query.feature_distances(&self.points[i]))
                } else {
                    None
                };
                NeighborDetail {
                    distance: dist,
                    output: self.outputs[i],
                    feature_distances,
                }
            })
            .collect();

        // Compute weighted mean prediction
        let prediction = if neighbors.is_empty() {
            f64::NAN
        } else {
            let mut exact = None;
            let mut ws = 0.0;
            let mut vs = 0.0;
            for n in &neighbors {
                if n.distance == 0.0 {
                    exact = Some(n.output);
                    break;
                }
                let w = 1.0 / n.distance;
                ws += w;
                vs += w * n.output;
            }
            exact.unwrap_or_else(|| if ws > 0.0 { vs / ws } else { f64::NAN })
        };

        PredictionDiagnostics {
            prediction,
            k,
            neighbors,
        }
    }

    fn compute_output_stats(&self) -> OutputStats {
        if self.is_empty() {
            return OutputStats {
                min: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
                num_distinct: 0,
            };
        }

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;
        let mut distinct: Vec<f64> = Vec::new();

        for &o in &self.outputs {
            min = min.min(o);
            max = max.max(o);
            sum += o;
            if !distinct.iter().any(|&v| (v - o).abs() < 1e-10) {
                distinct.push(o);
            }
        }

        OutputStats {
            min,
            max,
            mean: sum / self.outputs.len() as f64,
            num_distinct: distinct.len(),
        }
    }
}

impl LearnedMetric {
    /// Get per-feature diagnostics including effect curves.
    pub fn feature_diagnostics(&self) -> Vec<FeatureDiagnostics> {
        self.effect_regressions
            .iter()
            .zip(self.weights.iter())
            .enumerate()
            .map(|(i, (reg, &weight))| {
                let points = reg.get_points_sorted();
                let effect_curve: Vec<(f64, f64)> =
                    points.iter().map(|p| (*p.x(), *p.y())).collect();

                FeatureDiagnostics {
                    index: i,
                    weight,
                    effect_curve,
                }
            })
            .collect()
    }
}
