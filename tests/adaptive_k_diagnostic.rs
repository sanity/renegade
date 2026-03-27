//! Diagnostic: Is there learnable structure in per-point optimal k?
//!
//! For each LOO evaluation point, we compute the best k (lowest error).
//! Then we ask:
//!   1. Is there meaningful variance in best_k across points?
//!   2. If we used per-point oracle k, how much would accuracy improve?
//!   3. Can a secondary renegade model predict best_k from features?

use renegade_ml::{DataPoint, Renegade};

// --- Reuse the generic CSV point from diverse_datasets ---

#[derive(Clone, Debug)]
struct CsvPoint {
    values: Vec<f64>,
    ranges: Vec<(f64, f64)>,
}

impl DataPoint for CsvPoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        self.values
            .iter()
            .zip(other.values.iter())
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
        self.values.clone()
    }
}

/// Load CSV with target in specified column (0 = first, -1 or ncols-1 = last).
/// `target_col`: column index of the target. Use `None` for last column.
fn load_csv_target(data: &str, target_col: Option<usize>) -> Vec<(CsvPoint, f64)> {
    let rows: Vec<Vec<f64>> = data
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let vals: Result<Vec<f64>, _> = line.split(',').map(|v| v.trim().parse()).collect();
            vals.ok()
        })
        .collect();

    if rows.is_empty() {
        return Vec::new();
    }

    let ncols = rows[0].len();
    let tc = target_col.unwrap_or(ncols - 1);

    // Feature column indices = all columns except target
    let feat_cols: Vec<usize> = (0..ncols).filter(|&c| c != tc).collect();
    let nfeatures = feat_cols.len();

    let mut mins = vec![f64::MAX; nfeatures];
    let mut maxs = vec![f64::MIN; nfeatures];
    for row in &rows {
        for (fi, &ci) in feat_cols.iter().enumerate() {
            mins[fi] = mins[fi].min(row[ci]);
            maxs[fi] = maxs[fi].max(row[ci]);
        }
    }
    let ranges: Vec<(f64, f64)> = mins.into_iter().zip(maxs).collect();

    rows.into_iter()
        .map(|row| {
            let values: Vec<f64> = feat_cols.iter().map(|&ci| row[ci]).collect();
            let target = row[tc];
            (
                CsvPoint {
                    values,
                    ranges: ranges.clone(),
                },
                target,
            )
        })
        .collect()
}

fn load_csv(data: &str) -> Vec<(CsvPoint, f64)> {
    load_csv_target(data, None)
}

fn is_classification(data: &[(CsvPoint, f64)]) -> bool {
    data.iter().all(|(_, y)| (y - y.round()).abs() < 1e-6)
        && data
            .iter()
            .map(|(_, y)| y.round() as i64)
            .collect::<std::collections::HashSet<_>>()
            .len()
            <= 20
}

/// For each point in `data`, compute LOO error for each k and return the best k.
/// Also returns the error achieved by global-best-k and by per-point-oracle-k.
struct AdaptiveKAnalysis {
    /// Per-point best k values
    best_k_per_point: Vec<usize>,
    /// Global optimal k (single best across all points)
    global_k: usize,
    /// LOO error using global k
    global_k_error: f64,
    /// LOO error using per-point oracle k (upper bound on adaptive-k benefit)
    oracle_k_error: f64,
    /// LOO error using a renegade model to predict k
    predicted_k_error: f64,
    /// k values predicted by the secondary model
    predicted_k_per_point: Vec<usize>,
    /// Statistics on best_k distribution
    k_mean: f64,
    k_std: f64,
    k_min: usize,
    k_max: usize,
}

fn analyze_adaptive_k(data: &[(CsvPoint, f64)]) -> AdaptiveKAnalysis {
    let n = data.len();
    let max_k = ((n - 1) as f64).sqrt().ceil() as usize;
    let max_k = max_k.max(1).min(n - 1);
    let is_class = is_classification(data);

    // For each point, compute error at each k via LOO
    let mut best_k_per_point = Vec::with_capacity(n);
    let mut errors_per_point: Vec<Vec<f64>> = Vec::with_capacity(n); // [point][k] -> error

    for i in 0..n {
        // Build model without point i
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }

        // Compute distances from point i to all others
        let mut errors_for_k = vec![0.0f64; max_k + 1]; // index 0 unused

        // Get neighbors at max_k, then we can derive errors for all smaller k
        let neighbors = model.query_k(&data[i].0, max_k);
        let nb = &neighbors.neighbors;

        if is_class {
            // Classification: track class counts incrementally
            let mut counts: Vec<(f64, usize)> = Vec::new();
            for k in 1..=max_k.min(nb.len()) {
                let val = nb[k - 1].output;
                if let Some(entry) = counts.iter_mut().find(|(v, _)| (*v - val).abs() < 1e-10) {
                    entry.1 += 1;
                } else {
                    counts.push((val, 1));
                }
                let predicted = counts.iter().max_by_key(|(_, c)| *c).unwrap().0;
                errors_for_k[k] = if (predicted - data[i].1).abs() > 0.5 {
                    1.0
                } else {
                    0.0
                };
            }
        } else {
            // Regression: inverse-distance weighted mean
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            let mut has_exact = false;
            let mut exact_val = 0.0;

            for k in 1..=max_k.min(nb.len()) {
                let dist = nb[k - 1].distance;
                let out = nb[k - 1].output;

                if !has_exact {
                    if dist == 0.0 {
                        has_exact = true;
                        exact_val = out;
                    } else {
                        let w = 1.0 / dist;
                        weight_sum += w;
                        value_sum += w * out;
                    }
                }

                let predicted = if has_exact {
                    exact_val
                } else if weight_sum > 0.0 {
                    value_sum / weight_sum
                } else {
                    continue;
                };

                let err = predicted - data[i].1;
                errors_for_k[k] = err * err;
            }
        }

        // Find best k for this point (smallest k that achieves near-best error)
        let min_error = errors_for_k[1..=max_k]
            .iter()
            .copied()
            .fold(f64::MAX, f64::min);
        let epsilon = if is_class { 0.0 } else { min_error * 0.05 }; // 5% tolerance for regression
        let best_k = (1..=max_k)
            .find(|&k| errors_for_k[k] <= min_error + epsilon)
            .unwrap_or(1);

        best_k_per_point.push(best_k);
        errors_per_point.push(errors_for_k);
    }

    // Global optimal k: minimize total error across all points
    let mut global_errors = vec![0.0f64; max_k + 1];
    for errors in &errors_per_point {
        for k in 1..=max_k {
            global_errors[k] += errors[k];
        }
    }
    let global_k = (1..=max_k)
        .min_by(|&a, &b| {
            global_errors[a]
                .partial_cmp(&global_errors[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1);

    // Global k error (normalized)
    let global_k_error = global_errors[global_k] / n as f64;

    // Oracle k error: use each point's best k
    let oracle_k_error: f64 = (0..n)
        .map(|i| errors_per_point[i][best_k_per_point[i]])
        .sum::<f64>()
        / n as f64;

    // k distribution stats
    let k_mean = best_k_per_point.iter().sum::<usize>() as f64 / n as f64;
    let k_var = best_k_per_point
        .iter()
        .map(|&k| (k as f64 - k_mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let k_std = k_var.sqrt();
    let k_min = *best_k_per_point.iter().min().unwrap();
    let k_max = *best_k_per_point.iter().max().unwrap();

    // --- Can we predict k from features? ---
    // Train a secondary renegade model: features -> best_k
    // Use LOO to evaluate its predictions
    let mut predicted_k_per_point = Vec::with_capacity(n);
    let mut predicted_k_total_error = 0.0;

    for i in 0..n {
        let mut k_model = Renegade::new();
        for (j, (p, _)) in data.iter().enumerate() {
            if j != i {
                k_model.add(p.clone(), best_k_per_point[j] as f64);
            }
        }
        let predicted_k_raw = k_model.predict(&data[i].0);
        let predicted_k = predicted_k_raw.round().max(1.0).min(max_k as f64) as usize;
        predicted_k_per_point.push(predicted_k);

        // Evaluate: what error does the predicted k achieve for this point?
        predicted_k_total_error += errors_per_point[i][predicted_k];
    }
    let predicted_k_error = predicted_k_total_error / n as f64;

    AdaptiveKAnalysis {
        best_k_per_point,
        global_k,
        global_k_error,
        oracle_k_error,
        predicted_k_error,
        predicted_k_per_point,
        k_mean,
        k_std,
        k_min,
        k_max,
    }
}

fn print_analysis(name: &str, data: &[(CsvPoint, f64)], a: &AdaptiveKAnalysis) {
    let is_class = is_classification(data);
    let n = data.len();

    eprintln!("\n=== {} (n={}) ===", name, n);
    eprintln!(
        "  Global optimal k: {}  (LOO {})",
        a.global_k,
        if is_class {
            format!("error rate: {:.1}%", a.global_k_error * 100.0)
        } else {
            format!("RMSE: {:.4}", a.global_k_error.sqrt())
        }
    );
    eprintln!(
        "  Per-point best k: mean={:.1}, std={:.1}, range=[{}, {}]",
        a.k_mean, a.k_std, a.k_min, a.k_max
    );

    // Histogram of best k values
    let mut hist: Vec<(usize, usize)> = Vec::new();
    for &k in &a.best_k_per_point {
        if let Some(entry) = hist.iter_mut().find(|(v, _)| *v == k) {
            entry.1 += 1;
        } else {
            hist.push((k, 1));
        }
    }
    hist.sort_by_key(|(k, _)| *k);
    let top5: Vec<String> = hist
        .iter()
        .take(10)
        .map(|(k, c)| format!("k={}:{}", k, c))
        .collect();
    eprintln!("  k histogram (top 10): {}", top5.join(", "));

    eprintln!(
        "  Oracle k (per-point best): {}",
        if is_class {
            format!("error rate: {:.1}%", a.oracle_k_error * 100.0)
        } else {
            format!("RMSE: {:.4}", a.oracle_k_error.sqrt())
        }
    );
    eprintln!(
        "  Predicted k (renegade meta-model): {}",
        if is_class {
            format!("error rate: {:.1}%", a.predicted_k_error * 100.0)
        } else {
            format!("RMSE: {:.4}", a.predicted_k_error.sqrt())
        }
    );

    let improvement_oracle = if is_class {
        (a.global_k_error - a.oracle_k_error) * 100.0
    } else {
        a.global_k_error.sqrt() - a.oracle_k_error.sqrt()
    };
    let improvement_predicted = if is_class {
        (a.global_k_error - a.predicted_k_error) * 100.0
    } else {
        a.global_k_error.sqrt() - a.predicted_k_error.sqrt()
    };

    eprintln!(
        "  Headroom (oracle vs global): {}{:.4}",
        if improvement_oracle >= 0.0 { "+" } else { "" },
        improvement_oracle
    );
    eprintln!(
        "  Achieved (predicted vs global): {}{:.4}",
        if improvement_predicted >= 0.0 {
            "+"
        } else {
            ""
        },
        improvement_predicted
    );

    // Check: does the predicted k at least differ from global k?
    let using_global = a
        .predicted_k_per_point
        .iter()
        .filter(|&&k| k == a.global_k)
        .count();
    eprintln!(
        "  Predicted k == global k: {}/{} ({:.0}%)",
        using_global,
        n,
        using_global as f64 / n as f64 * 100.0
    );
}

#[test]
fn adaptive_k_iris() {
    let data = load_csv(include_str!("../testdata/iris.csv"));
    let a = analyze_adaptive_k(&data);
    print_analysis("Iris", &data, &a);
}

#[test]
fn adaptive_k_wine() {
    let data = load_csv_target(include_str!("../testdata/wine.csv"), Some(0));
    let a = analyze_adaptive_k(&data);
    print_analysis("Wine", &data, &a);
}

#[test]
fn adaptive_k_auto_mpg() {
    let data = load_csv_target(include_str!("../testdata/auto_mpg.csv"), Some(0));
    let a = analyze_adaptive_k(&data);
    print_analysis("Auto MPG", &data, &a);
}

#[test]
fn adaptive_k_breast_cancer() {
    let data = load_csv(include_str!("../testdata/breast_cancer.csv"));
    let a = analyze_adaptive_k(&data);
    print_analysis("Breast Cancer", &data, &a);
}

#[test]
fn adaptive_k_ionosphere() {
    let data = load_csv(include_str!("../testdata/ionosphere.csv"));
    let a = analyze_adaptive_k(&data);
    print_analysis("Ionosphere", &data, &a);
}
