use crate::dispersion::Dispersion;
use crate::predict::ExtrapolatedPrediction;

/// A single nearest neighbor result.
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// Distance from query point (0 = identical).
    pub distance: f64,
    /// Output value of this training point.
    pub output: f64,
    /// Instance weight (default 1.0). Higher weight means this point
    /// has more influence on predictions.
    pub weight: f64,
}

/// A set of nearest neighbors, sorted by distance.
#[derive(Debug, Clone)]
pub struct Neighbors {
    pub neighbors: Vec<Neighbor>,
}

impl Neighbors {
    /// Weighted average of neighbor outputs.
    /// Combines inverse-distance weighting with instance weights:
    /// effective_weight = instance_weight / distance.
    pub fn weighted_mean(&self) -> f64 {
        if self.neighbors.is_empty() {
            return f64::NAN;
        }

        // If any neighbor has distance 0, return weighted average of exact matches.
        let exact: Vec<&Neighbor> = self
            .neighbors
            .iter()
            .filter(|n| n.distance == 0.0)
            .collect();
        if !exact.is_empty() {
            let total_w: f64 = exact.iter().map(|n| n.weight).sum();
            if total_w > 0.0 {
                return exact.iter().map(|n| n.weight * n.output).sum::<f64>() / total_w;
            }
            return exact[0].output;
        }

        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for n in &self.neighbors {
            let w = n.weight / n.distance;
            weight_sum += w;
            value_sum += w * n.output;
        }
        value_sum / weight_sum
    }

    /// Gaussian kernel weighted average: w(d) = instance_weight * exp(-d²/(2h²)).
    /// Unlike hard-k + 1/d, this gives smooth decay — distant neighbors contribute
    /// proportionally less without a sharp cutoff.
    pub fn gaussian_weighted_mean(&self, bandwidth: f64) -> f64 {
        if self.neighbors.is_empty() {
            return f64::NAN;
        }

        // Exact matches: same handling as weighted_mean
        let exact: Vec<&Neighbor> = self
            .neighbors
            .iter()
            .filter(|n| n.distance == 0.0)
            .collect();
        if !exact.is_empty() {
            let total_w: f64 = exact.iter().map(|n| n.weight).sum();
            if total_w > 0.0 {
                return exact.iter().map(|n| n.weight * n.output).sum::<f64>() / total_w;
            }
            return exact[0].output;
        }

        let h2 = 2.0 * bandwidth * bandwidth;
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for n in &self.neighbors {
            let w = (-n.distance * n.distance / h2).exp() * n.weight;
            if w < 1e-15 {
                // Beyond ~6 sigma, negligible contribution — stop early
                // since neighbors are sorted by distance
                break;
            }
            weight_sum += w;
            value_sum += w * n.output;
        }
        if weight_sum > 0.0 {
            value_sum / weight_sum
        } else {
            // Bandwidth too small for any neighbor to contribute — fall back to nearest
            self.neighbors[0].output
        }
    }

    /// Extrapolate output to distance=0 by fitting a linear trend.
    pub fn extrapolate(&self) -> ExtrapolatedPrediction {
        ExtrapolatedPrediction::from_neighbors(&self.neighbors)
    }

    /// `(weight, output)` pairs matching `weighted_mean()`'s own weighting:
    /// exact matches only (`instance_weight`) if any neighbor is at
    /// distance 0, otherwise every neighbor weighted by
    /// `instance_weight / distance`.
    fn mean_weight_pairs(&self) -> Vec<(f64, f64)> {
        let exact: Vec<&Neighbor> = self
            .neighbors
            .iter()
            .filter(|n| n.distance == 0.0)
            .collect();
        if !exact.is_empty() {
            exact.iter().map(|n| (n.weight, n.output)).collect()
        } else {
            self.neighbors
                .iter()
                .map(|n| (n.weight / n.distance, n.output))
                .collect()
        }
    }

    /// Dispersion of neighbor outputs about `weighted_mean()`, computed
    /// with the identical weighting (see `mean_weight_pairs`) so the spread
    /// reported here always describes the estimate `weighted_mean()`
    /// actually returns.
    ///
    /// Returns `None` for an empty neighbor set — there is no mean to
    /// disperse around.
    pub fn dispersion(&self) -> Option<Dispersion> {
        if self.neighbors.is_empty() {
            return None;
        }
        Some(Dispersion::from_weighted_pairs(&self.mean_weight_pairs()))
    }

    /// `(weight, output)` pairs matching `gaussian_weighted_mean(bandwidth)`'s
    /// own weighting: exact matches only (`instance_weight`) if any
    /// neighbor is at distance 0, otherwise
    /// `instance_weight * exp(-distance²/(2*bandwidth²))` for each
    /// neighbor, stopping at the same negligible-weight cutoff used there
    /// (neighbors are sorted by distance, so once a weight underflows, all
    /// later ones do too). Returns `None` when no neighbor contributes a
    /// non-negligible weight — the same degenerate case where
    /// `gaussian_weighted_mean` falls back to the nearest neighbor's output
    /// alone.
    fn gaussian_weight_pairs(&self, bandwidth: f64) -> Option<Vec<(f64, f64)>> {
        if self.neighbors.is_empty() {
            return None;
        }

        let exact: Vec<&Neighbor> = self
            .neighbors
            .iter()
            .filter(|n| n.distance == 0.0)
            .collect();
        if !exact.is_empty() {
            return Some(exact.iter().map(|n| (n.weight, n.output)).collect());
        }

        let h2 = 2.0 * bandwidth * bandwidth;
        let mut pairs = Vec::new();
        for n in &self.neighbors {
            let w = (-n.distance * n.distance / h2).exp() * n.weight;
            if w < 1e-15 {
                // Beyond ~6 sigma, negligible contribution — stop early,
                // matching gaussian_weighted_mean's early exit.
                break;
            }
            pairs.push((w, n.output));
        }
        if pairs.is_empty() {
            None
        } else {
            Some(pairs)
        }
    }

    /// Dispersion of neighbor outputs about
    /// `gaussian_weighted_mean(bandwidth)`, computed with the identical
    /// weighting (see `gaussian_weight_pairs`).
    ///
    /// Unlike `dispersion()`'s unbounded inverse-distance kernel, the
    /// Gaussian kernel is bounded in `[0, 1]` and decays to ~0 beyond a few
    /// bandwidths, so the resulting `Dispersion::weight_sum` — the kernel
    /// mass — is a meaningful "is there evidence near this query" signal
    /// (see [`Dispersion::weight_sum`]).
    ///
    /// Returns `None` for an empty neighbor set, or when `bandwidth` is so
    /// small that no neighbor contributes non-negligible weight — the same
    /// degenerate case where `gaussian_weighted_mean` falls back to the
    /// nearest neighbor's output alone, for which no population exists to
    /// report dispersion over.
    pub fn gaussian_dispersion(&self, bandwidth: f64) -> Option<Dispersion> {
        let pairs = self.gaussian_weight_pairs(bandwidth)?;
        Some(Dispersion::from_weighted_pairs(&pairs))
    }

    /// Class probabilities: weighted fraction of neighbors with each distinct output value.
    /// Combines inverse-distance weighting with instance weights.
    pub fn class_votes(&self) -> Vec<(f64, f64)> {
        if self.neighbors.is_empty() {
            return Vec::new();
        }

        let mut counts: Vec<(f64, f64)> = Vec::new(); // (class, total_weight)
        for n in &self.neighbors {
            let w = if n.distance == 0.0 {
                n.weight * 1e6 // very large but finite weight for exact matches
            } else {
                n.weight / n.distance
            };
            if let Some(entry) = counts
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += w;
            } else {
                counts.push((n.output, w));
            }
        }

        let total: f64 = counts.iter().map(|(_, w)| w).sum();
        let n_classes = counts.len() as f64;
        if total > 0.0 {
            counts
                .into_iter()
                .map(|(class, w)| (class, w / total))
                .collect()
        } else {
            counts
                .into_iter()
                .map(|(class, _)| (class, 1.0 / n_classes))
                .collect()
        }
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
