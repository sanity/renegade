//! Profile the Gaussian kernel path: training overhead and inference cost.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use renegade_ml::{DataPoint, Renegade};
use std::time::Instant;

#[derive(Clone, Debug)]
struct ProfilePoint {
    features: Vec<f64>,
    ranges: Vec<(f64, f64)>,
}

impl DataPoint for ProfilePoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        self.features
            .iter()
            .zip(other.features.iter())
            .zip(self.ranges.iter())
            .map(|((a, b), (lo, hi))| {
                let range = hi - lo;
                if range == 0.0 {
                    0.0
                } else {
                    (a - b).abs() / range
                }
            })
            .collect()
    }

    fn feature_values(&self) -> Vec<f64> {
        self.features.clone()
    }
}

/// Regression dataset (continuous output) so Gaussian kernel gets activated.
fn make_regression_dataset(n: usize, d: usize, seed: u64) -> Vec<(ProfilePoint, f64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ranges = vec![(0.0, 1.0); d];
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| rng.gen()).collect();
            // Non-integer output to ensure regression detection
            let output: f64 = features.iter().take(3.min(d)).sum::<f64>() + rng.gen::<f64>() * 0.1;
            (
                ProfilePoint {
                    features,
                    ranges: ranges.clone(),
                },
                output,
            )
        })
        .collect()
}

/// Classification dataset (integer output) — Gaussian kernel should NOT activate.
fn make_classification_dataset(n: usize, d: usize, seed: u64) -> Vec<(ProfilePoint, f64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ranges = vec![(0.0, 1.0); d];
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| rng.gen()).collect();
            let output = if features[0] > 0.5 { 1.0 } else { 0.0 };
            (
                ProfilePoint {
                    features,
                    ranges: ranges.clone(),
                },
                output,
            )
        })
        .collect()
}

#[test]
fn profile_gaussian_training_overhead() {
    eprintln!();
    eprintln!("=== Gaussian Kernel Training Overhead ===");
    eprintln!(
        "{:<8} {:<5} {:<12} {:<18} {:<18} {:<10} {:<10}",
        "n", "d", "type", "train (ms)", "of which bw (ms)", "bw active", "bw value"
    );
    eprintln!("{}", "-".repeat(85));

    for &(n, d) in &[
        (100, 5),
        (500, 5),
        (1000, 5),
        (1000, 20),
        (5000, 5),
        (10000, 5),
    ] {
        // Regression (Gaussian kernel candidate)
        {
            let data = make_regression_dataset(n, d, 42);
            let mut model = Renegade::new();
            for (p, o) in &data {
                model.add(p.clone(), *o);
            }
            let t0 = Instant::now();
            let _ = model.predict(&data[0].0);
            let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let diag = model.diagnostics();

            eprintln!(
                "{:<8} {:<5} {:<12} {:<18.1} {:<18} {:<10} {:<10}",
                n,
                d,
                "regression",
                train_ms,
                "-", // Can't separate yet without instrumenting
                diag.kernel_bandwidth.is_some(),
                diag.kernel_bandwidth
                    .map(|h| format!("{:.4}", h))
                    .unwrap_or("-".into()),
            );
        }

        // Classification (should skip Gaussian)
        {
            let data = make_classification_dataset(n, d, 42);
            let mut model = Renegade::new();
            for (p, o) in &data {
                model.add(p.clone(), *o);
            }
            let t0 = Instant::now();
            let _ = model.predict(&data[0].0);
            let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let diag = model.diagnostics();

            eprintln!(
                "{:<8} {:<5} {:<12} {:<18.1} {:<18} {:<10} {:<10}",
                n,
                d,
                "classif.",
                train_ms,
                "skipped",
                diag.kernel_bandwidth.is_some(),
                "-",
            );
        }
    }
    eprintln!();
}

#[test]
fn profile_gaussian_inference() {
    eprintln!();
    eprintln!("=== Gaussian Kernel Inference Cost ===");
    eprintln!(
        "{:<8} {:<5} {:<18} {:<18} {:<18}",
        "n", "d", "predict_k(5) µs", "predict() µs", "predict() method"
    );
    eprintln!("{}", "-".repeat(75));

    for &(n, d) in &[(100, 5), (500, 5), (1000, 5), (5000, 5), (10000, 5)] {
        let data = make_regression_dataset(n, d, 42);
        let mut model = Renegade::new();
        for (p, o) in &data {
            model.add(p.clone(), *o);
        }
        // Force training
        let _ = model.predict(&data[0].0);
        let diag = model.diagnostics();
        let method = if diag.kernel_bandwidth.is_some() {
            "gaussian"
        } else {
            "hard-k"
        };

        let iters = 1000;
        let query = &data[n / 2].0;

        // Warm up
        for _ in 0..100 {
            let _ = model.predict_k(query, 5);
            let _ = model.predict(query);
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = model.predict_k(query, 5);
        }
        let pk_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = model.predict(query);
        }
        let p_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        eprintln!(
            "{:<8} {:<5} {:<18.1} {:<18.1} {:<18}",
            n, d, pk_us, p_us, method
        );
    }
    eprintln!();
}
