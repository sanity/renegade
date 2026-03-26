use crate::predict::ExtrapolatedPrediction;

/// A single nearest neighbor result.
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// Distance from query point (0 = identical).
    pub distance: f64,
    /// Output value of this training point.
    pub output: f64,
}

/// A set of nearest neighbors, sorted by distance.
#[derive(Debug, Clone)]
pub struct Neighbors {
    pub neighbors: Vec<Neighbor>,
}

impl Neighbors {
    /// Simple weighted average of neighbor outputs (inverse distance weighting).
    pub fn weighted_mean(&self) -> f64 {
        if self.neighbors.is_empty() {
            return f64::NAN;
        }

        // If any neighbor has distance 0, return its output (exact match).
        for n in &self.neighbors {
            if n.distance == 0.0 {
                return n.output;
            }
        }

        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for n in &self.neighbors {
            let w = 1.0 / n.distance;
            weight_sum += w;
            value_sum += w * n.output;
        }
        value_sum / weight_sum
    }

    /// Extrapolate output to distance=0 by fitting a linear trend.
    pub fn extrapolate(&self) -> ExtrapolatedPrediction {
        ExtrapolatedPrediction::from_neighbors(&self.neighbors)
    }

    /// Class probabilities: fraction of neighbors with each distinct output value.
    /// Useful for classification where outputs are class labels encoded as integers.
    pub fn class_votes(&self) -> Vec<(f64, f64)> {
        if self.neighbors.is_empty() {
            return Vec::new();
        }

        let mut counts: Vec<(f64, usize)> = Vec::new();
        for n in &self.neighbors {
            if let Some(entry) = counts
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += 1;
            } else {
                counts.push((n.output, 1));
            }
        }

        let total = self.neighbors.len() as f64;
        counts
            .into_iter()
            .map(|(class, count)| (class, count as f64 / total))
            .collect()
    }

    /// Random sample from neighbors (uniform).
    pub fn sample(&self, rng_value: f64) -> Option<f64> {
        if self.neighbors.is_empty() {
            return None;
        }
        let idx = (rng_value * self.neighbors.len() as f64) as usize;
        let idx = idx.min(self.neighbors.len() - 1);
        Some(self.neighbors[idx].output)
    }
}
