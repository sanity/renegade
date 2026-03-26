use pav_regression::{IsotonicRegression, Point as PavPoint};

/// A learned distance metric that transforms features into "effect space" before
/// computing distances.
///
/// For each feature, fits a univariate isotonic regression: feature_value → output.
/// This learns each feature's marginal effect on the output. Distances are then
/// computed in this effect space, where features that predict the output well
/// contribute more to distance, and noise features (flat effect) contribute nothing.
///
/// This avoids the sign-cancellation problem of pair-wise distance learning:
/// the isotonic regressions learn point-wise effects, not pair-wise distances.
pub struct LearnedMetric {
    /// One isotonic regression per feature: maps raw feature value → effect on output.
    effect_regressions: Vec<IsotonicRegression<f64>>,
    /// Weight per feature: proportion of output variance explained by this feature's effect.
    weights: Vec<f64>,
}

/// Training data for metric learning: raw feature values and output.
pub struct TrainingPoint {
    pub features: Vec<f64>,
    pub output: f64,
}

impl LearnedMetric {
    /// Learn a metric from training data.
    ///
    /// For each feature, fits an ascending isotonic regression from feature value to output.
    /// Then computes feature weights based on how much output variance each feature explains.
    pub fn learn(points: &[TrainingPoint]) -> Self {
        if points.is_empty() {
            return LearnedMetric {
                effect_regressions: Vec::new(),
                weights: Vec::new(),
            };
        }

        let num_features = points[0].features.len();
        if num_features == 0 {
            return LearnedMetric {
                effect_regressions: Vec::new(),
                weights: Vec::new(),
            };
        }

        // For each feature, try both ascending and descending isotonic regression.
        // Keep whichever explains more variance (higher R²).
        let mut effect_regressions = Vec::with_capacity(num_features);
        let mut variances_explained = Vec::with_capacity(num_features);

        let mean_y: f64 = points.iter().map(|p| p.output).sum::<f64>() / points.len() as f64;
        let ss_tot: f64 = points.iter().map(|p| (p.output - mean_y).powi(2)).sum();

        for f in 0..num_features {
            let pav_points: Vec<PavPoint<f64>> = points
                .iter()
                .map(|p| PavPoint::new(p.features[f], p.output))
                .collect();

            // Try ascending
            let asc = IsotonicRegression::new_ascending(&pav_points);
            // Try descending
            let desc = IsotonicRegression::new_descending(&pav_points);

            let (best_reg, best_var) = match (asc, desc) {
                (Ok(a), Ok(d)) => {
                    let var_a = variance_explained(&a, points, f, ss_tot);
                    let var_d = variance_explained(&d, points, f, ss_tot);
                    if var_a >= var_d {
                        (a, var_a)
                    } else {
                        (d, var_d)
                    }
                }
                (Ok(a), Err(_)) => {
                    let var_a = variance_explained(&a, points, f, ss_tot);
                    (a, var_a)
                }
                (Err(_), Ok(d)) => {
                    let var_d = variance_explained(&d, points, f, ss_tot);
                    (d, var_d)
                }
                (Err(_), Err(_)) => {
                    // Fallback: trivial regression
                    let trivial = IsotonicRegression::new_ascending(&[
                        PavPoint::new(0.0, mean_y),
                        PavPoint::new(1.0, mean_y),
                    ])
                    .expect("trivial regression should never fail");
                    (trivial, 0.0)
                }
            };

            effect_regressions.push(best_reg);
            variances_explained.push(best_var.max(0.0));
        }

        // Compute weights: proportion of variance explained, normalized.
        let total_var: f64 = variances_explained.iter().sum();
        let weights = if total_var > 0.0 {
            variances_explained.iter().map(|v| v / total_var).collect()
        } else {
            // All features equally uninformative — equal weights
            vec![1.0 / num_features as f64; num_features]
        };

        LearnedMetric {
            effect_regressions,
            weights,
        }
    }

    /// Compute distance between two points using the learned metric.
    ///
    /// Transforms each feature value through its effect regression, then computes
    /// a weighted Manhattan distance in effect space.
    ///
    /// `features_a` and `features_b` are the raw feature values (not distances).
    pub fn distance(&self, features_a: &[f64], features_b: &[f64]) -> f64 {
        if self.effect_regressions.is_empty() {
            return f64::NAN;
        }

        self.effect_regressions
            .iter()
            .zip(self.weights.iter())
            .enumerate()
            .map(|(i, (reg, &w))| {
                let effect_a = reg.interpolate(features_a[i]).unwrap_or(0.0);
                let effect_b = reg.interpolate(features_b[i]).unwrap_or(0.0);
                w * (effect_a - effect_b).abs()
            })
            .sum()
    }

    /// Number of features this metric was trained on.
    pub fn num_features(&self) -> usize {
        self.effect_regressions.len()
    }

    /// Get the learned weight for each feature (for diagnostics).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

/// Compute fraction of output variance explained by this feature's isotonic regression.
fn variance_explained(
    reg: &IsotonicRegression<f64>,
    points: &[TrainingPoint],
    feature_idx: usize,
    ss_tot: f64,
) -> f64 {
    if ss_tot == 0.0 {
        return 0.0;
    }
    let ss_res: f64 = points
        .iter()
        .map(|p| {
            let predicted = reg.interpolate(p.features[feature_idx]).unwrap_or(0.0);
            (p.output - predicted).powi(2)
        })
        .sum();
    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_feature_gets_higher_weight_than_noise() {
        // Feature 0: perfectly predicts output (y = f0)
        // Feature 1: random noise
        let points: Vec<TrainingPoint> = (0..100)
            .map(|i| {
                let f0 = i as f64 / 100.0;
                let f1 = ((i * 37) % 100) as f64 / 100.0;
                TrainingPoint {
                    features: vec![f0, f1],
                    output: f0,
                }
            })
            .collect();

        let metric = LearnedMetric::learn(&points);

        assert!(
            metric.weights()[0] > metric.weights()[1] * 5.0,
            "Signal feature should have much higher weight: w0={:.3}, w1={:.3}",
            metric.weights()[0],
            metric.weights()[1]
        );
    }

    #[test]
    fn equal_additive_features_get_similar_weights() {
        // y = f0 + f1 + f2, all features equally predictive
        let points: Vec<TrainingPoint> = (0..200)
            .map(|i| {
                let f0 = (i % 10) as f64 / 10.0;
                let f1 = ((i / 10) % 10) as f64 / 10.0;
                let f2 = ((i * 7) % 10) as f64 / 10.0;
                TrainingPoint {
                    features: vec![f0, f1, f2],
                    output: f0 + f1 + f2,
                }
            })
            .collect();

        let metric = LearnedMetric::learn(&points);

        eprintln!(
            "Equal additive weights: [{:.3}, {:.3}, {:.3}]",
            metric.weights()[0],
            metric.weights()[1],
            metric.weights()[2]
        );

        // All weights should be roughly equal (within 3x)
        let max_w = metric.weights().iter().cloned().fold(0.0f64, f64::max);
        let min_w = metric.weights().iter().cloned().fold(1.0f64, f64::min);
        assert!(
            max_w < min_w * 3.0,
            "Weights should be roughly equal: {:?}",
            metric.weights()
        );
    }

    #[test]
    fn noise_features_get_near_zero_weight() {
        // y = f0, features 1-4 are noise
        let points: Vec<TrainingPoint> = (0..100)
            .map(|i| {
                let f0 = i as f64 / 100.0;
                TrainingPoint {
                    features: vec![
                        f0,
                        ((i * 37) % 100) as f64 / 100.0,
                        ((i * 53) % 100) as f64 / 100.0,
                        ((i * 71) % 100) as f64 / 100.0,
                        ((i * 89) % 100) as f64 / 100.0,
                    ],
                    output: f0,
                }
            })
            .collect();

        let metric = LearnedMetric::learn(&points);

        eprintln!("1 signal + 4 noise weights: {:?}", metric.weights());

        // Feature 0 should dominate
        assert!(
            metric.weights()[0] > 0.5,
            "Signal feature should have >50% weight, got {:.3}",
            metric.weights()[0]
        );
    }
}
