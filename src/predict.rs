use crate::neighbor::Neighbor;

/// Result of extrapolating neighbor outputs to distance=0.
#[derive(Debug, Clone)]
pub struct ExtrapolatedPrediction {
    /// Predicted output at distance=0.
    pub value: f64,
    /// R² of the linear fit. Low values mean the trend is unreliable.
    pub r_squared: f64,
    /// Number of neighbors used.
    pub k: usize,
}

impl ExtrapolatedPrediction {
    pub(crate) fn from_neighbors(neighbors: &[Neighbor]) -> Self {
        let k = neighbors.len();

        if k == 0 {
            return ExtrapolatedPrediction {
                value: f64::NAN,
                r_squared: 0.0,
                k: 0,
            };
        }

        if k == 1 {
            return ExtrapolatedPrediction {
                value: neighbors[0].output,
                r_squared: 0.0,
                k: 1,
            };
        }

        // Linear regression: output = a + b * distance
        // Extrapolated prediction = a (intercept at distance=0)
        let n = k as f64;
        let sum_x: f64 = neighbors.iter().map(|n| n.distance).sum();
        let sum_y: f64 = neighbors.iter().map(|n| n.output).sum();
        let sum_xy: f64 = neighbors.iter().map(|n| n.distance * n.output).sum();
        let sum_xx: f64 = neighbors.iter().map(|n| n.distance * n.distance).sum();

        let denom = n * sum_xx - sum_x * sum_x;

        if denom.abs() < 1e-15 {
            // All neighbors at the same distance — just average.
            return ExtrapolatedPrediction {
                value: sum_y / n,
                r_squared: 0.0,
                k,
            };
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;

        // R² calculation
        let mean_y = sum_y / n;
        let ss_tot: f64 = neighbors.iter().map(|n| (n.output - mean_y).powi(2)).sum();
        let ss_res: f64 = neighbors
            .iter()
            .map(|n| {
                let predicted = a + b * n.distance;
                (n.output - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            1.0 // All outputs identical — perfect "fit"
        };

        ExtrapolatedPrediction {
            value: a,
            r_squared,
            k,
        }
    }
}
