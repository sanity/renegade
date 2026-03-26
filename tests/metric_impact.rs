use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use renegade::{DataPoint, Renegade};

/// Generic numeric point with arbitrary feature count.
#[derive(Clone, Debug)]
struct NumericPoint {
    features: Vec<f64>,
    ranges: Vec<(f64, f64)>,
}

impl DataPoint for NumericPoint {
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

    fn num_features(&self) -> usize {
        self.features.len()
    }
}

struct SyntheticResult {
    name: &'static str,
    n: usize,
    num_features: usize,
    no_metric_k5: f64,
    no_metric_auto_k: (f64, usize),
    with_metric_auto_k: (f64, usize),
}

impl SyntheticResult {
    fn print(&self) {
        eprintln!(
            "{:<35} n={:<4} f={:<3} | no-metric k=5: {:.3} | no-metric auto k={}: {:.3} | metric auto k={}: {:.3} | metric delta: {:+.3}",
            self.name,
            self.n,
            self.num_features,
            self.no_metric_k5,
            self.no_metric_auto_k.1,
            self.no_metric_auto_k.0,
            self.with_metric_auto_k.1,
            self.with_metric_auto_k.0,
            self.with_metric_auto_k.0 - self.no_metric_auto_k.0,
        );
    }
}

/// LOO evaluation. `use_metric` controls whether the learned metric is active.
/// Returns (error, k_used).
/// For classification: error = misclassification rate.
/// For regression: error = RMSE.
fn loo_eval(
    data: &[(NumericPoint, f64)],
    is_classification: bool,
    use_metric: bool,
    fixed_k: Option<usize>,
) -> (f64, usize) {
    let n = data.len();

    // Get K: either fixed, or auto-determined
    let k = if let Some(k) = fixed_k {
        k
    } else {
        let mut full_model = Renegade::new();
        for (point, output) in data {
            full_model.add(point.clone(), *output);
        }
        if use_metric {
            full_model.get_optimal_k() // triggers metric learning + K selection
        } else {
            // Compute K without metric by using query_k in LOO
            let max_k = (n as f64).sqrt().ceil() as usize;
            let max_k = max_k.max(1).min(n - 1);
            let mut best_k = 1;
            let mut best_err = f64::MAX;
            for try_k in 1..=max_k {
                let err = loo_raw(data, try_k, is_classification);
                if err < best_err {
                    best_err = err;
                    best_k = try_k;
                }
            }
            best_k
        }
    };

    let err = if use_metric {
        // Build model with metric, then evaluate
        loo_with_metric(data, k, is_classification)
    } else {
        loo_raw(data, k, is_classification)
    };

    (err, k)
}

/// LOO without learned metric (simple mean distance).
fn loo_raw(data: &[(NumericPoint, f64)], k: usize, is_classification: bool) -> f64 {
    let n = data.len();
    let mut total_error = 0.0;

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (point, output)) in data.iter().enumerate() {
            if j != i {
                model.add(point.clone(), *output);
            }
        }

        let neighbors = model.query_k(&data[i].0, k);

        if is_classification {
            let votes = neighbors.class_votes();
            let predicted = votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            if (predicted - data[i].1).abs() > 0.5 {
                total_error += 1.0;
            }
        } else {
            let predicted = neighbors.weighted_mean();
            let err = predicted - data[i].1;
            total_error += err * err;
        }
    }

    if is_classification {
        total_error / n as f64 // misclassification rate
    } else {
        (total_error / n as f64).sqrt() // RMSE
    }
}

/// LOO with learned metric. Learns metric on full data, uses it for LOO eval.
fn loo_with_metric(data: &[(NumericPoint, f64)], k: usize, is_classification: bool) -> f64 {
    let n = data.len();

    // Learn metric on full dataset (same as what auto does)
    let mut full_model = Renegade::new();
    for (point, output) in data {
        full_model.add(point.clone(), *output);
    }
    full_model.get_optimal_k(); // triggers metric learning

    // Now evaluate LOO using that model's query_k (which uses the learned metric)
    let mut total_error = 0.0;
    for i in 0..n {
        // Query from full model — the point itself will be nearest (dist=0),
        // so we ask for k+1 and skip the first.
        let neighbors = full_model.query_k(&data[i].0, k + 1);
        let filtered: Vec<_> = neighbors
            .neighbors
            .iter()
            .filter(|n| {
                n.distance > 0.0 || {
                    // Keep at most k non-self neighbors
                    true
                }
            })
            .skip(1) // skip self-match
            .take(k)
            .collect();

        if is_classification {
            let mut counts: Vec<(f64, usize)> = Vec::new();
            for n in &filtered {
                if let Some(entry) = counts
                    .iter_mut()
                    .find(|(v, _)| (*v - n.output).abs() < 1e-10)
                {
                    entry.1 += 1;
                } else {
                    counts.push((n.output, 1));
                }
            }
            if let Some(predicted) = counts.iter().max_by_key(|(_, c)| *c) {
                if (predicted.0 - data[i].1).abs() > 0.5 {
                    total_error += 1.0;
                }
            }
        } else {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            let mut exact = None;
            for n in &filtered {
                if n.distance == 0.0 {
                    exact = Some(n.output);
                    break;
                }
                let w = 1.0 / n.distance;
                weight_sum += w;
                value_sum += w * n.output;
            }
            let predicted = exact.unwrap_or_else(|| {
                if weight_sum > 0.0 {
                    value_sum / weight_sum
                } else {
                    0.0
                }
            });
            let err = predicted - data[i].1;
            total_error += err * err;
        }
    }

    if is_classification {
        total_error / n as f64
    } else {
        (total_error / n as f64).sqrt()
    }
}

fn make_point(features: Vec<f64>, ranges: &[(f64, f64)]) -> NumericPoint {
    NumericPoint {
        features,
        ranges: ranges.to_vec(),
    }
}

fn eval_dataset(
    name: &'static str,
    data: Vec<(NumericPoint, f64)>,
    is_classification: bool,
) -> SyntheticResult {
    let n = data.len();
    let num_features = data[0].0.num_features();
    let (no_metric_k5, _) = loo_eval(&data, is_classification, false, Some(5));
    let no_metric_auto_k = loo_eval(&data, is_classification, false, None);
    let with_metric_auto_k = loo_eval(&data, is_classification, true, None);

    SyntheticResult {
        name,
        n,
        num_features,
        no_metric_k5,
        no_metric_auto_k,
        with_metric_auto_k,
    }
}

#[test]
fn metric_impact_comparison() {
    let mut results = Vec::new();

    // 1. Pure signal, no noise — metric shouldn't hurt
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let ranges = vec![(0.0, 1.0); 3];
        let data: Vec<_> = (0..100)
            .map(|_| {
                let f: Vec<f64> = (0..3).map(|_| rng.gen()).collect();
                let out = f[0] + f[1] + f[2];
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("3 features, all signal", data, false));
    }

    // 2. Signal + noise features — metric should help
    {
        let mut rng = SmallRng::seed_from_u64(2);
        let ranges = vec![(0.0, 1.0); 6];
        let data: Vec<_> = (0..100)
            .map(|_| {
                let f: Vec<f64> = (0..6).map(|_| rng.gen()).collect();
                let out = f[0] + f[1] + f[2]; // only first 3 matter
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("3 signal + 3 noise", data, false));
    }

    // 3. Mostly noise — metric should help a lot
    {
        let mut rng = SmallRng::seed_from_u64(3);
        let ranges = vec![(0.0, 1.0); 10];
        let data: Vec<_> = (0..100)
            .map(|_| {
                let f: Vec<f64> = (0..10).map(|_| rng.gen()).collect();
                let out = f[0]; // only feature 0 matters
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("1 signal + 9 noise", data, false));
    }

    // 4. Classification with noise features
    {
        let mut rng = SmallRng::seed_from_u64(4);
        let ranges = vec![(0.0, 1.0); 6];
        let data: Vec<_> = (0..120)
            .map(|_| {
                let f: Vec<f64> = (0..6).map(|_| rng.gen()).collect();
                let class = if f[0] + f[1] > 1.0 { 1.0 } else { 0.0 }; // first 2 features
                (make_point(f, &ranges), class)
            })
            .collect();
        results.push(eval_dataset("2-class, 2 signal + 4 noise", data, true));
    }

    // 5. Small dataset, high dimensionality — curse of dimensionality
    {
        let mut rng = SmallRng::seed_from_u64(5);
        let ranges = vec![(0.0, 1.0); 15];
        let data: Vec<_> = (0..50)
            .map(|_| {
                let f: Vec<f64> = (0..15).map(|_| rng.gen()).collect();
                let out = f[0] + f[1]; // only 2 of 15 features matter
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("2 signal + 13 noise, n=50", data, false));
    }

    // 6. All features equally predictive — metric shouldn't change much
    {
        let mut rng = SmallRng::seed_from_u64(6);
        let ranges = vec![(0.0, 1.0); 5];
        let data: Vec<_> = (0..100)
            .map(|_| {
                let f: Vec<f64> = (0..5).map(|_| rng.gen()).collect();
                let out: f64 = f.iter().sum();
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("5 features, all equal signal", data, false));
    }

    // 7. Nonlinear relationship
    {
        let mut rng = SmallRng::seed_from_u64(7);
        let ranges = vec![(0.0, 1.0); 4];
        let data: Vec<_> = (0..100)
            .map(|_| {
                let f: Vec<f64> = (0..4).map(|_| rng.gen()).collect();
                let out = (f[0] * f[1]) + f[2].powi(2); // nonlinear, feature 3 is noise
                (make_point(f, &ranges), out)
            })
            .collect();
        results.push(eval_dataset("nonlinear, 3 signal + 1 noise", data, false));
    }

    eprintln!();
    eprintln!("=== Metric Impact Comparison ===");
    eprintln!(
        "{:<35} {:<10} {:<5} | {:<20} | {:<25} | {:<25} | {}",
        "Dataset", "n", "f", "no-metric k=5", "no-metric auto-K", "with-metric auto-K", "delta"
    );
    eprintln!("{}", "-".repeat(160));
    for r in &results {
        r.print();
    }
    eprintln!();
}
