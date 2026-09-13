//! Durable evidence for the decision documented on
//! `Renegade::predict_with_prior`: shrinking `predict()` toward the
//! dataset's global mean BY DEFAULT was evaluated and rejected, because on
//! a direct reconstruction of the motivating scenario it made predictions
//! measurably worse -- especially in the signal-bearing region shrinkage
//! was specifically meant to help.
//!
//! None of this crate's other bundled datasets (iris/wine/auto_mpg/
//! ionosphere/breast_cancer/wine_quality) have this shape: ~92% of queries
//! have no real local signal (true target ~0, outputs are pure noise
//! around it), ~8% sit in a genuinely learnable localized region. This is a
//! synthetic reconstruction of that pattern (originally the freenet-core
//! routing-prediction scenario that motivated adding shrinkage at all).
//!
//! Kept as a committed, rerunnable test (rather than the throwaway script
//! that originally produced these numbers) per review feedback: evidence
//! backing a permanent public-API design decision should be reproducible
//! from the repo, not just narrated in a doc comment.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use renegade_ml::{DataPoint, Renegade};

#[derive(Clone, Debug)]
struct Point1D {
    x: f64,
}

impl DataPoint for Point1D {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        vec![(self.x - other.x).abs()]
    }
    fn feature_values(&self) -> Vec<f64> {
        vec![self.x]
    }
}

/// `n` points uniform in [0, 1]. Points with x in `signal_region` (an ~8%
/// wide band) get true value `signal_value` plus noise; everything else
/// gets true value 0.0 plus the same noise.
fn generate(
    n: usize,
    signal_region: (f64, f64),
    signal_value: f64,
    seed: u64,
) -> Vec<(Point1D, f64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.gen_range(0.0..1.0);
            let in_signal = x >= signal_region.0 && x < signal_region.1;
            let true_value = if in_signal { signal_value } else { 0.0 };
            let noise: f64 = rng.gen_range(-1.0..1.0);
            (Point1D { x }, true_value + noise)
        })
        .collect()
}

/// LOO comparison of predict() (raw) against predict_with_prior(query,
/// global_output_mean()) (shrunk, using the crate's SHIPPED
/// local_signal_variance/shrink_toward machinery -- not a reimplementation),
/// split into overall and signal-region MSE.
fn loo_compare(data: &[(Point1D, f64)], signal_region: (f64, f64)) -> (f64, f64, f64, f64) {
    let n = data.len();
    let mut sse_raw_overall = 0.0;
    let mut sse_shrunk_overall = 0.0;
    let mut sse_raw_targeted = 0.0;
    let mut sse_shrunk_targeted = 0.0;
    let mut n_targeted = 0usize;

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let raw = model.predict(&data[i].0);
        let prior = model.global_output_mean().unwrap();
        let shrunk = model.predict_with_prior(&data[i].0, prior);

        let actual = data[i].1;
        let raw_err = raw - actual;
        let shrunk_err = shrunk - actual;
        sse_raw_overall += raw_err * raw_err;
        sse_shrunk_overall += shrunk_err * shrunk_err;

        let x = data[i].0.x;
        if x >= signal_region.0 && x < signal_region.1 {
            n_targeted += 1;
            sse_raw_targeted += raw_err * raw_err;
            sse_shrunk_targeted += shrunk_err * shrunk_err;
        }
    }

    (
        sse_raw_overall / n as f64,
        sse_shrunk_overall / n as f64,
        sse_raw_targeted / n_targeted.max(1) as f64,
        sse_shrunk_targeted / n_targeted.max(1) as f64,
    )
}

#[test]
fn shrink_by_default_hurts_the_signal_region_it_was_meant_to_help() {
    // 8% of the domain (0.60..0.68) carries a real, learnable offset of 10;
    // the remaining 92% has true value 0 -- both regimes share the same
    // noise distribution.
    let region = (0.60, 0.68);
    let data = generate(300, region, 10.0, 42);

    let (mse_raw_overall, mse_shrunk_overall, mse_raw_targeted, mse_shrunk_targeted) =
        loo_compare(&data, region);

    eprintln!("=== Synthetic 92%-flat / 8%-signal regression (seed=42) ===");
    eprintln!("  raw    overall MSE: {mse_raw_overall:.4}   targeted MSE: {mse_raw_targeted:.4}");
    eprintln!(
        "  shrunk overall MSE: {mse_shrunk_overall:.4}   targeted MSE: {mse_shrunk_targeted:.4}"
    );
    eprintln!(
        "  overall change: {:+.1}%   targeted change: {:+.1}%",
        100.0 * (mse_shrunk_overall - mse_raw_overall) / mse_raw_overall,
        100.0 * (mse_shrunk_targeted - mse_raw_targeted) / mse_raw_targeted
    );

    // The mechanism (documented on Renegade::local_signal_variance and
    // Renegade::predict_with_prior): a neighborhood straddling the
    // boundary between the signal region and the flat background has HIGH
    // local variance -- a mix of two true regimes, not noise -- which
    // local_signal_variance misreads as "no local signal, shrink hard",
    // discarding an already-reasonable local mean. This is the consistent,
    // decision-relevant finding (see the review discussion referenced in
    // predict_with_prior's doc comment for why overall MSE alone is a
    // noisier signal than the signal-region MSE specifically).
    assert!(
        mse_shrunk_targeted > mse_raw_targeted,
        "expected shrink-by-default to hurt the signal-bearing region (documented finding): raw={mse_raw_targeted:.4} shrunk={mse_shrunk_targeted:.4}"
    );
}

#[test]
fn shrink_by_default_signal_region_harm_is_consistent_across_seeds() {
    // Same shape as above, swept across several seeds to confirm the
    // signal-region degradation isn't an artifact of one particular noise
    // draw (per review feedback: this was independently verified across 7
    // seeds during review, confirming direction in 7/7 for the shipped
    // subtraction-based local_signal_variance formula).
    let region = (0.60, 0.68);
    for seed in [1u64, 2, 3, 4, 5, 42, 999_999] {
        let data = generate(300, region, 10.0, seed);
        let (_, _, mse_raw_targeted, mse_shrunk_targeted) = loo_compare(&data, region);
        assert!(
            mse_shrunk_targeted > mse_raw_targeted,
            "seed {seed}: expected shrink-by-default to hurt the signal region, got raw={mse_raw_targeted:.4} shrunk={mse_shrunk_targeted:.4}"
        );
    }
}
