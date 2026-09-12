/// Answers "do these neighbors agree?" — a companion to a weighted mean.
///
/// A point estimate alone can't distinguish a neighborhood of tightly
/// agreeing observations from one where a few outliers and a pile of
/// unrelated noise happen to average out to the same number. `Dispersion`
/// reports the spread of neighbor outputs using the SAME weights as the
/// mean it accompanies (see [`crate::Neighbors::dispersion`] and
/// [`crate::Neighbors::gaussian_dispersion`]), so the two can never
/// describe different populations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dispersion {
    /// The weighted mean these statistics are computed about. Equal to
    /// `weighted_mean()` (or `gaussian_weighted_mean(bandwidth)`) for the
    /// same neighbor set.
    pub mean: f64,
    /// Weighted population variance of neighbor outputs about `mean`:
    /// `Σw(x - mean)² / Σw`, using the same weights as `mean`. Zero when
    /// every contributing neighbor agrees exactly.
    pub variance: f64,
    /// `sqrt(variance)`.
    pub std_dev: f64,
    /// Kish's effective sample size: `(Σw)² / Σw²`. This is
    /// scale-invariant — k neighbors of any uniform weight report
    /// `effective_n ≈ k` regardless of what that weight is, so k neighbors
    /// at a uniform distance report ≈k no matter how far away that distance
    /// is. It answers "how many roughly-independent observations back this
    /// estimate", NOT "is there evidence near this query" — a large
    /// `effective_n` built entirely from distant neighbors looks identical
    /// to one built from close ones. Use `weight_sum` for a quantity that,
    /// for a bounded kernel, distinguishes those.
    pub effective_n: f64,
    /// `Σw`: the total weight mass behind the estimate, using the same
    /// weights as `mean`. For a kernel bounded in `[0, 1]` that decays with
    /// distance (e.g. the Gaussian kernel behind
    /// [`crate::Neighbors::gaussian_dispersion`]), this is a meaningful
    /// "is there evidence near this query" signal: it shrinks toward 0 as
    /// the query moves away from all training data. For the unbounded
    /// inverse-distance kernel behind [`crate::Neighbors::dispersion`]
    /// (`weight = instance_weight / distance`, which grows without bound as
    /// distance approaches 0 and never reaches 0 as distance grows), it
    /// does not have this property — prefer `effective_n` there.
    pub weight_sum: f64,
}

impl Dispersion {
    /// Compute dispersion statistics from `(weight, output)` pairs.
    /// `pairs` must be non-empty — callers filter empty/degenerate cases
    /// (no neighbors, or no neighbor contributing non-negligible weight)
    /// before reaching here and return `None` instead.
    pub(crate) fn from_weighted_pairs(pairs: &[(f64, f64)]) -> Self {
        debug_assert!(
            !pairs.is_empty(),
            "Dispersion::from_weighted_pairs requires at least one pair"
        );

        let weight_sum: f64 = pairs.iter().map(|(w, _)| w).sum();

        if weight_sum <= 0.0 {
            // Degenerate: no positive weight to average over (e.g. every
            // instance weight is zero or negative, which violates the
            // crate's own "weight must be positive" contract but is
            // enforced only by debug_assert in add_weighted). Fall back to
            // the first pair alone, mirroring weighted_mean()'s own
            // fallback in the equivalent situation.
            return Dispersion {
                mean: pairs[0].1,
                variance: 0.0,
                std_dev: 0.0,
                effective_n: 1.0,
                weight_sum,
            };
        }

        let mean = pairs.iter().map(|(w, o)| w * o).sum::<f64>() / weight_sum;
        let variance = pairs
            .iter()
            .map(|(w, o)| w * (o - mean) * (o - mean))
            .sum::<f64>()
            / weight_sum;
        let sum_w2: f64 = pairs.iter().map(|(w, _)| w * w).sum();
        let effective_n = if sum_w2 > 0.0 {
            weight_sum * weight_sum / sum_w2
        } else {
            0.0
        };

        Dispersion {
            mean,
            variance,
            std_dev: variance.sqrt(),
            effective_n,
            weight_sum,
        }
    }
}
