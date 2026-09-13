mod diagnostics;
mod dispersion;
mod metric;
mod neighbor;
mod predict;
/// Vantage-point tree for metric-space nearest neighbor search.
pub mod vptree;

pub use diagnostics::{
    FeatureDiagnostics, ModelDiagnostics, NeighborDetail, OutputStats, PredictionDiagnostics,
};
pub use dispersion::{Dispersion, Shrinkage};
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

    // --- Running weighted variance for `global_output_variance` ---
    // Maintained incrementally on `add_weighted` (O(1) per point) and
    // recomputed from scratch on `retain` (already O(n) there) via
    // `accumulate_output`/`recompute_output_sums`. Kept separate from the
    // per-query `Dispersion` machinery in dispersion.rs: these describe the
    // WHOLE dataset, not one neighbor set, and a query may run once per
    // routing decision, so rescanning every stored point on every call would
    // undo the amortized-training design this crate otherwise commits to.
    //
    // This uses West's incremental weighted variance (a weighted Welford's
    // algorithm: track a running mean and update it before folding each new
    // point into the second moment) rather than the more obvious "track
    // Σw, Σw·o, Σw·o², derive variance as E[o²] − E[o]²" — that one-pass
    // formula suffers catastrophic cancellation whenever the outputs share a
    // large common offset relative to their true spread (e.g. two outputs
    // 1e8 and 1e8+1 have variance 0.25, but E[o²]−E[o]² can round to 0, or
    // for larger offsets to an arbitrarily wrong LARGE positive number — not
    // bounded by "a few ULPs negative", so `.max(0.0)` does not save it).
    // West's algorithm has no such cancellation regardless of magnitude,
    // while remaining exactly O(1) amortized per point.
    //
    // It has a DIFFERENT, narrower failure mode instead: the incremental
    // mean update (`delta = output - output_mean`, see `accumulate_output`)
    // can itself overflow f64 for outputs of opposite sign each individually
    // within roughly a factor of 2 of f64::MAX (e.g. 1e308 and -1e308 have a
    // perfectly representable true mean of 0.0, but computing their
    // difference during the update overflows to -Infinity, so
    // `global_output_mean()`/`global_output_variance()` report an infinite
    // rather than the true finite answer). This is a boundary of f64
    // representability, not specific to West's algorithm — any accumulator
    // needs SOME subtraction or sum of the raw values, and no realistic
    // regression target (routing metrics, distances, latencies, ...) comes
    // remotely close to 1e308 in magnitude. Documented here rather than
    // "fixed" with extended-precision arithmetic, since that complexity
    // isn't proportionate to inputs this extreme — but note the failure is
    // at least visible (±Infinity, not a plausible-looking wrong finite
    // number) rather than silent.
    /// Σ instance_weight, over all stored points.
    output_weight_sum: f64,
    /// Running weighted mean of all stored outputs.
    output_mean: f64,
    /// Running weighted second moment about `output_mean` (West's
    /// algorithm). `output_m2 / output_weight_sum` is the population
    /// variance.
    output_m2: f64,

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
            output_weight_sum: 0.0,
            output_mean: 0.0,
            output_m2: 0.0,
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
        self.accumulate_output(weight, output);

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
        self.recompute_output_sums();
        self.invalidate();
    }

    /// Fold one more `(weight, output)` pair into the running weighted mean
    /// and second moment (West's incremental algorithm — see the field docs
    /// on `output_weight_sum` for why).
    fn accumulate_output(&mut self, weight: f64, output: f64) {
        self.output_weight_sum += weight;
        let delta = output - self.output_mean;
        self.output_mean += (weight / self.output_weight_sum) * delta;
        let delta2 = output - self.output_mean;
        self.output_m2 += weight * delta * delta2;
    }

    /// Recompute the running weighted mean/second-moment from scratch. Only
    /// needed after a bulk removal (`retain`) — `add_weighted` maintains
    /// them incrementally via `accumulate_output` since it only ever adds,
    /// and West's algorithm has no way to "remove" a point from a running
    /// mean/second-moment pair without redoing the fold.
    fn recompute_output_sums(&mut self) {
        self.output_weight_sum = 0.0;
        self.output_mean = 0.0;
        self.output_m2 = 0.0;
        // Can't iterate-and-mutate via `zip` directly on `self`'s own
        // fields; collect nothing extra though — just re-borrow per index.
        for i in 0..self.outputs.len() {
            let (w, o) = (self.instance_weights[i], self.outputs[i]);
            self.accumulate_output(w, o);
        }
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
    ///
    /// This is the RAW local estimate — it does not shrink toward the
    /// dataset's global mean. See [`Self::predict_with_prior`] for a
    /// shrunk alternative, and its doc comment for why shrinkage is opt-in
    /// here rather than `predict()`'s default (measured to make the
    /// crate's own accuracy benchmarks worse, not better, in the general
    /// case — see that method's docs for the numbers).
    pub fn predict(&mut self, query: &P) -> f64 {
        let (neighbors, bandwidth) = self.query_for_predict(query);
        match bandwidth {
            Some(h) => neighbors.gaussian_weighted_mean(h),
            None => neighbors.weighted_mean(),
        }
    }

    /// Predict output using automatically determined K, shrunk toward a
    /// caller-supplied `prior` by how much local evidence there is to trust:
    ///
    /// ```text
    /// estimate = prior + λ·(local_mean − prior)
    /// ```
    ///
    /// using the same [`Dispersion::shrink_toward`]/[`Self::local_signal_variance`]
    /// building blocks [`Self::shrink`] is built on — not a call to
    /// `shrink()` itself, since `shrink()` only supports the
    /// inverse-distance `dispersion()` path, while this also needs to
    /// handle the Gaussian-kernel path `predict()` may have selected.
    ///
    /// **This was evaluated as `predict()`'s new DEFAULT and rejected** —
    /// worth recording here since the obvious next question is "why is this
    /// opt-in rather than automatic". Measured (with this exact shrinkage —
    /// `local_signal_variance` — against `predict()`'s raw output) on this
    /// crate's own real-dataset LOO benchmarks, and against a reconstruction
    /// of the exact motivating scenario (a mostly-flat regression target
    /// with a small, genuinely learnable localized region). All of the
    /// following are committed, rerunnable tests, not one-off numbers —
    /// see `tests/gaussian_kernel_bench.rs` (`*_predict_with_prior_vs_raw`)
    /// and `tests/shrinkage_by_default_evidence.rs`:
    ///
    /// - **auto_mpg** (real regression dataset): shrink-by-default RMSE
    ///   2.5809 vs. raw 2.5746 — a small but real regression.
    /// - **wine_quality** (real regression dataset): shrink-by-default RMSE
    ///   0.7191 vs. raw 0.7537 — a genuine ~4.6% improvement. Real datasets
    ///   give a genuinely MIXED picture, not a uniform regression.
    /// - **Synthetic 92%-flat/8%-signal reconstruction**: shrink-by-default
    ///   made the signal-region MSE dramatically worse (14.21 vs. 1.47 raw,
    ///   one seed; confirmed worse across 7 seeds in
    ///   `shrink_by_default_signal_region_harm_is_consistent_across_seeds`)
    ///   — the opposite of the intended effect on the exact case this was
    ///   meant to fix. Overall MSE was also worse for this seed (1.61 vs.
    ///   0.59 raw) but is a noisier signal across seeds/formulas than the
    ///   signal-region figure — the signal-region harm is the robust,
    ///   decision-relevant finding, not "every metric always gets worse".
    ///
    /// The mechanism: `local_signal_variance` treats "this neighborhood's
    /// own variance is unusually HIGH relative to the dataset overall" as
    /// evidence of "no local signal, shrink hard" — but a neighborhood
    /// straddling the BOUNDARY between a real signal region and the flat
    /// background also has high local variance (a mix of two true regimes,
    /// not noise), and gets shrunk just as hard, discarding a local mean
    /// that was already a reasonable (if imperfect) estimate. This is not a
    /// tuning issue — it's a structural property of using local-vs-global
    /// variance comparison as a trust signal: it cannot distinguish "no
    /// structure here" from "structure that changes sharply right here",
    /// and the latter is arguably the case where a routing decision matters
    /// most.
    ///
    /// Given a wrong default is worse than no default, shrinkage stays
    /// opt-in. Use this method when you've evaluated it on your own data
    /// and confirmed it helps rather than hurts.
    pub fn predict_with_prior(&mut self, query: &P, prior: f64) -> f64 {
        let (neighbors, bandwidth) = self.query_for_predict(query);
        self.shrink_local_mean(&neighbors, bandwidth, prior)
    }

    /// Shared setup for `predict`/`predict_with_prior`: train if needed,
    /// then fetch the neighbor set (and kernel bandwidth, if any) predict()
    /// has always used.
    fn query_for_predict(&mut self, query: &P) -> (Neighbors, Option<f64>) {
        self.ensure_trained();
        let k = self.optimal_k.unwrap();
        let bandwidth = self.kernel_bandwidth;
        let neighbors = if let Some(_h) = bandwidth {
            // Gaussian kernel: query max_k neighbors so the kernel has a full
            // neighborhood to weight. The kernel itself does the "soft cutoff" —
            // distant neighbors contribute exponentially less.
            let max_k = (self.len() as f64).sqrt().ceil() as usize;
            self.query_k(query, max_k)
        } else {
            self.query_k(query, k)
        };
        (neighbors, bandwidth)
    }

    /// Compute the (possibly Gaussian-kernel) local mean for `neighbors`,
    /// then shrink it toward `prior` using `local_signal_variance` for the
    /// signal-variance term. Falls back to the plain local mean whenever
    /// there's no dispersion to shrink with (empty neighbors, or a Gaussian
    /// bandwidth too small for any neighbor to contribute) or no global
    /// variance to compare against (no training data — shouldn't occur via
    /// `predict_with_prior`, which trains first, but this keeps the
    /// fallback explicit rather than panicking).
    fn shrink_local_mean(&self, neighbors: &Neighbors, bandwidth: Option<f64>, prior: f64) -> f64 {
        let (local_mean, dispersion) = match bandwidth {
            Some(h) => (
                neighbors.gaussian_weighted_mean(h),
                neighbors.gaussian_dispersion(h),
            ),
            None => (neighbors.weighted_mean(), neighbors.dispersion()),
        };
        let Some(dispersion) = dispersion else {
            return local_mean;
        };
        let Some(signal_variance) = self.local_signal_variance(&dispersion) else {
            return local_mean;
        };
        dispersion.shrink_toward(prior, signal_variance).estimate
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

    /// Weighted mean of every stored output — the running mean maintained
    /// internally by the same West's-algorithm accumulator that backs
    /// [`Self::global_output_variance`], exposed directly since it's already
    /// computed as a byproduct. A natural choice of `prior` for
    /// [`Self::predict_with_prior`] when the caller has no better one, though
    /// `predict()` itself does NOT use this as a default — see
    /// `predict_with_prior`'s doc comment for why shrinkage stays opt-in.
    ///
    /// `None` under the same degenerate conditions as
    /// `global_output_variance` (no data, or every instance weight
    /// non-positive). Can return `Some(an infinite value)` for outputs of
    /// opposite sign each individually near `f64::MAX` — see the caveat on
    /// the `output_weight_sum` field.
    pub fn global_output_mean(&self) -> Option<f64> {
        if self.output_weight_sum <= 0.0 {
            None
        } else {
            Some(self.output_mean)
        }
    }

    /// Weighted variance of every stored output, treating the whole training
    /// set as one population — `Σw(o - mean)² / Σw` over every point ever
    /// added (and still present after any `retain`). This is the crate's
    /// only "global" statistic; everything else (`Dispersion`, `Neighbors`)
    /// describes one query's neighborhood.
    ///
    /// `None` if there is no data, or if every instance weight is
    /// non-positive (the same degenerate case `Dispersion` falls back on —
    /// see its `from_weighted_pairs`).
    ///
    /// A NaN or ±infinite stored output poisons this permanently (every
    /// later call also reports NaN) rather than being silently discarded —
    /// see the field docs on `output_weight_sum` for why this is computed
    /// via West's algorithm instead of the more obvious "derive variance
    /// from Σw/Σw·o/Σw·o²" formula, which both loses precision AND would
    /// silently launder a NaN result to a confident-looking `Some(0.0)`
    /// via a naive `.max(0.0)` clamp. Extreme-magnitude (near `f64::MAX`)
    /// opposite-sign stored outputs have a separate, narrower overflow
    /// boundary — see the field docs on `output_weight_sum`.
    pub fn global_output_variance(&self) -> Option<f64> {
        if self.output_weight_sum <= 0.0 {
            return None;
        }
        let variance = self.output_m2 / self.output_weight_sum;
        // Clamp tiny float noise to 0 (variance is mathematically >= 0),
        // but only for an actually-finite result — `f64::max` silently
        // picks the non-NaN operand, so `NaN.max(0.0) == 0.0`. Checking
        // `is_nan()` first keeps a NaN (from a NaN/±Infinity stored output)
        // visibly NaN instead of laundering it into a false "zero variance".
        Some(if variance.is_nan() {
            variance
        } else {
            variance.max(0.0)
        })
    }

    /// Estimate the between-neighborhood ("signal") variance of the target
    /// near a query, for use as `signal_variance` in
    /// [`Dispersion::shrink_toward`].
    ///
    /// Decomposes the GLOBAL variance of every stored output into a LOCAL
    /// component — `local.variance`, the given neighborhood's own dispersion,
    /// treated as noise — and whatever variance is left over, attributed to
    /// genuine local signal:
    ///
    /// ```text
    /// signal_variance ≈ max(0, global_output_variance() − local.variance)
    /// ```
    ///
    /// Rationale: a neighborhood whose outputs agree about as tightly as the
    /// dataset overall (`local.variance ≈ global_output_variance()`) has
    /// demonstrated no more structure than noise alone would produce —
    /// signal ≈ 0, so `shrink_toward` shrinks hard toward the prior. A
    /// neighborhood that agrees far more tightly than the dataset overall
    /// (`local.variance` well below the global figure) has captured
    /// something real; signal stays close to the global variance, so
    /// `shrink_toward` keeps trusting the local mean.
    ///
    /// This is deliberately a PER-QUERY estimate, not a single global
    /// constant. A single global "how much does the signal vary" number gets
    /// inflated by any strongly localized effect elsewhere in the dataset —
    /// a query sitting in a flat, no-signal region would still inherit that
    /// inflated figure and keep `shrink_toward`'s λ high (trusting a noisy
    /// local mean) exactly where it shouldn't. Comparing THIS neighborhood's
    /// dispersion against the global figure, instead of using the global
    /// figure alone, is what fixes that.
    ///
    /// Caveats — read before trusting this as a calibrated variance:
    ///
    /// - A neighbor set alone cannot distinguish "this neighborhood has low
    ///   true variation" from "these particular k points happen to agree by
    ///   chance". This is a method-of-moments point estimate (loosely the
    ///   same subtraction a one-way ANOVA or a DerSimonian-Laird
    ///   random-effects meta-analysis uses to split total variance into
    ///   between- and within-group components, though those aggregate
    ///   within-group variance across ALL groups — this substitutes a
    ///   single neighborhood's own variance instead), not a hypothesis
    ///   test, and it is noisiest exactly when `local.effective_n` is
    ///   small — the same regime where [`Dispersion::standard_error`] is
    ///   least trustworthy.
    /// - It also assumes noise is roughly homoskedastic across
    ///   neighborhoods. A neighborhood with a genuinely (not just by luck)
    ///   lower noise floor than the dataset's average will have its signal
    ///   systematically overestimated — this fixes the "one global constant
    ///   inflated by other neighborhoods" failure mode described above, but
    ///   does not fully separate signal from noise in general.
    /// - `local` should come from THIS model's own `Neighbors::dispersion()`
    ///   / `gaussian_dispersion()` — a `Dispersion` from elsewhere (or one
    ///   hand-constructed with a negative `variance`, since its fields are
    ///   public) is not clamped against and can produce a nonsensical
    ///   result.
    ///
    /// Treat the result as a heuristic prior for shrinkage, not a calibrated
    /// quantity.
    ///
    /// Returns `None` if there's no global variance to compare against (no
    /// training data, or every instance weight non-positive). Propagates
    /// NaN (rather than silently clamping it to 0) if `global_output_variance()`
    /// or `local.variance` is NaN.
    pub fn local_signal_variance(&self, local: &Dispersion) -> Option<f64> {
        let global_variance = self.global_output_variance()?;
        let signal_variance = global_variance - local.variance;
        Some(if signal_variance.is_nan() {
            signal_variance
        } else {
            signal_variance.max(0.0)
        })
    }

    /// Convenience: shrink `neighbors.weighted_mean()` toward `prior`, using
    /// this model's own local/global variance decomposition
    /// ([`local_signal_variance`](Self::local_signal_variance)) as the
    /// signal variance behind the shrinkage. This is the recommended entry
    /// point for most callers — it wires together `Neighbors::dispersion`,
    /// `local_signal_variance`, and `Dispersion::shrink_toward` with a
    /// consistent, correct choice of signal variance, rather than each
    /// caller re-deriving (and, empirically, mis-deriving) the same formula.
    ///
    /// Uses `neighbors.dispersion()` (the inverse-distance kernel matching
    /// `weighted_mean()`), not `gaussian_dispersion` — call
    /// `Dispersion::shrink_toward` directly if the Gaussian-kernel path is
    /// what your `Neighbors` was built for.
    ///
    /// Returns `None` if `neighbors` is empty, or there is no training data
    /// (or only non-positive instance weights) to compare against.
    pub fn shrink(&self, neighbors: &Neighbors, prior: f64) -> Option<Shrinkage> {
        let local = neighbors.dispersion()?;
        let signal_variance = self.local_signal_variance(&local)?;
        Some(local.shrink_toward(prior, signal_variance))
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
