/// Answers "do these neighbors agree?" — a companion to a weighted mean.
///
/// A point estimate alone can't distinguish a neighborhood of tightly
/// agreeing observations from one where a few outliers and a pile of
/// unrelated noise happen to average out to the same number. `Dispersion`
/// reports the spread of neighbor outputs using the SAME weights as the
/// mean it accompanies (see [`crate::Neighbors::dispersion`] and
/// [`crate::Neighbors::gaussian_dispersion`]), so the two can never
/// describe different populations.
#[derive(Debug, Clone)]
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
        // Kish's ESS, (Σw)²/Σw², is invariant to scaling every weight by the
        // same positive constant — so compute it on weights normalized by
        // their max instead of the raw weights. Mathematically identical,
        // but avoids `w * w` overflowing to infinity (and the ratio
        // collapsing to NaN) for legal large instance weights, since w² can
        // overflow f64 long before w or Σw does.
        let max_w = pairs.iter().fold(0.0_f64, |acc, (w, _)| acc.max(*w));
        let effective_n = if max_w > 0.0 {
            let scaled_sum: f64 = pairs.iter().map(|(w, _)| w / max_w).sum();
            let scaled_sum_w2: f64 = pairs.iter().map(|(w, _)| (w / max_w).powi(2)).sum();
            if scaled_sum_w2 > 0.0 {
                scaled_sum * scaled_sum / scaled_sum_w2
            } else {
                0.0
            }
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

    /// Standard error of `mean`: `sqrt(variance / effective_n)`.
    ///
    /// Uses `effective_n` (Kish's effective sample size), never `weight_sum`
    /// — this answers "how precisely is `mean` pinned down by roughly this
    /// many independent-ish observations", which is `effective_n`'s job by
    /// definition (see its doc comment). `weight_sum` is raw kernel mass;
    /// dividing by it would conflate "how much evidence is nearby" with "how
    /// many independent samples does that evidence represent", which for an
    /// unbounded kernel (`dispersion()`) or a large uniform weight are not
    /// the same number at all.
    ///
    /// Caveat: this is a plug-in estimator computed entirely from the
    /// neighbors it was built from, so it inherits their limits. With a
    /// single neighbor (or several neighbors that all agree exactly),
    /// `variance` is exactly zero by construction — there is nothing in the
    /// neighbor set itself to measure spread against — so `standard_error()`
    /// reports 0 (maximal confidence) rather than reflecting the real
    /// uncertainty of estimating a mean from few observations. Callers that
    /// need a noise floor for small neighborhoods should combine this with a
    /// separate estimate (e.g. [`crate::Renegade::local_signal_variance`]'s
    /// global comparison) rather than trusting `standard_error()` alone at
    /// low `effective_n`.
    pub fn standard_error(&self) -> f64 {
        (self.variance / self.effective_n).sqrt()
    }

    /// Empirical-Bayes (James-Stein form) shrinkage of `mean` toward a
    /// caller-supplied `prior`:
    ///
    /// ```text
    /// estimate = prior + λ · (mean − prior)
    /// λ = signal_variance / (signal_variance + standard_error()²)
    /// ```
    ///
    /// `prior` is domain knowledge the crate has no way to know — a global
    /// mean, a baseline rate, the model's own global prediction, whatever
    /// the caller would fall back to with zero local evidence. `signal_variance`
    /// is the between-neighborhood variance of the TRUE target: how much
    /// legitimate local signal is there to trust, as opposed to noise.
    /// [`crate::Renegade::local_signal_variance`] estimates it per-query from
    /// the model's own data; see its docs for why a single global constant
    /// is the wrong shape for this — it means using a domain-specific prior.
    ///
    /// λ is the fraction of the gap between `prior` and `mean` that survives:
    /// λ → 1 as the local estimate gets more precise (`standard_error` → 0)
    /// or the neighborhood carries more real signal (`signal_variance` grows)
    /// — trust the local mean fully. λ → 0 as the local estimate gets noisier
    /// or the neighborhood carries no more signal than chance would produce
    /// — fall back to the prior.
    ///
    /// `signal_variance` must be non-negative; NaN propagates (as does a NaN
    /// or infinite `prior`, `mean`, or `standard_error`), a negative
    /// non-NaN value is clamped to 0 (fully shrink to the prior — treated as
    /// "no local signal detected" rather than an error). When both
    /// `signal_variance` and `standard_error()` are exactly zero — no signal
    /// estimate AND no measured spread (e.g. a single exact-match neighbor)
    /// — there is nothing to distinguish trusting the local mean from
    /// trusting the prior; this degenerate case defaults to λ = 1 (trust the
    /// local observation), matching how `Dispersion` itself treats a single
    /// pair as its own whole population.
    pub fn shrink_toward(&self, prior: f64, signal_variance: f64) -> Shrinkage {
        let signal_variance = if signal_variance.is_nan() {
            signal_variance
        } else {
            signal_variance.max(0.0)
        };
        let standard_error = self.standard_error();
        let denom = signal_variance + standard_error * standard_error;
        let lambda = if denom > 0.0 {
            (signal_variance / denom).clamp(0.0, 1.0)
        } else if denom == 0.0 {
            1.0
        } else {
            // Unreachable for finite non-NaN inputs (both terms are
            // non-negative), so only a NaN operand lands here.
            f64::NAN
        };

        Shrinkage {
            estimate: prior + lambda * (self.mean - prior),
            lambda,
            standard_error,
        }
    }
}

/// Result of [`Dispersion::shrink_toward`]: a local mean pulled toward a
/// prior by an amount that depends on how much the local evidence is worth
/// trusting.
#[derive(Debug, Clone)]
pub struct Shrinkage {
    /// `prior + lambda * (mean - prior)`.
    pub estimate: f64,
    /// Shrinkage weight in `[0, 1]`. `1.0` = fully trust the local mean,
    /// `0.0` = fully fall back to the prior.
    pub lambda: f64,
    /// `Dispersion::standard_error()` of the local mean being shrunk —
    /// carried through so callers can see the precision behind `lambda`
    /// without recomputing it.
    pub standard_error: f64,
}
