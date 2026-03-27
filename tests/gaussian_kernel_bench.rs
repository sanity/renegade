//! Benchmark comparing predict() (which may use Gaussian kernel) vs predict_k()
//! (which always uses hard-k + 1/d) on regression datasets.

use renegade_ml::{DataPoint, Renegade};

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

/// LOO regression RMSE using predict() (auto k, auto metric, auto kernel).
fn loo_predict_rmse(data: &[(CsvPoint, f64)]) -> (f64, Option<f64>, usize) {
    let n = data.len();
    let mut sum_sq_err = 0.0;

    // Get diagnostics from full model to report what was selected
    let mut full_model = Renegade::new();
    for (p, y) in data {
        full_model.add(p.clone(), *y);
    }
    let _ = full_model.predict(&data[0].0); // force training
    let diag = full_model.diagnostics();
    let bandwidth = diag.kernel_bandwidth;
    let k = diag.optimal_k.unwrap_or(0);

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let predicted = model.predict(&data[i].0);
        let err = predicted - data[i].1;
        sum_sq_err += err * err;
    }

    ((sum_sq_err / n as f64).sqrt(), bandwidth, k)
}

/// LOO regression RMSE using predict_k() (hard k, no kernel).
fn loo_hard_k_rmse(data: &[(CsvPoint, f64)], k: usize) -> f64 {
    let n = data.len();
    let mut sum_sq_err = 0.0;

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let predicted = model.predict_k(&data[i].0, k);
        let err = predicted - data[i].1;
        sum_sq_err += err * err;
    }

    (sum_sq_err / n as f64).sqrt()
}

#[test]
fn auto_mpg_gaussian_vs_hard_k() {
    let data = load_csv_target(include_str!("../testdata/auto_mpg.csv"), Some(0));
    let mean_mpg: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;

    let (rmse_predict, bandwidth, auto_k) = loo_predict_rmse(&data);
    let rmse_k5 = loo_hard_k_rmse(&data, 5);

    eprintln!("=== Auto MPG: Gaussian Kernel vs Hard K ===");
    eprintln!("  Mean MPG: {:.1}", mean_mpg);
    eprintln!("  Auto-selected k={}, bandwidth={:?}", auto_k, bandwidth);
    eprintln!("  predict() RMSE (auto):   {:.4}", rmse_predict);
    eprintln!("  predict_k(5) RMSE:       {:.4}", rmse_k5);
    eprintln!("  Improvement: {:.4}", rmse_k5 - rmse_predict);

    // Should still pass the existing threshold
    assert!(
        rmse_predict < 4.5,
        "Auto MPG RMSE {:.2} above 4.5",
        rmse_predict
    );
}

#[test]
fn wine_quality_gaussian_vs_hard_k() {
    let data = load_csv_target(include_str!("../testdata/wine_quality.csv"), None);

    let mean_y: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;
    let std_y: f64 =
        (data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

    // Train/test split (LOO too slow for 4898 points)
    let split = (data.len() as f64 * 0.8) as usize;
    let (train, test) = data.split_at(split);

    // Build model with predict() path
    let mut model = Renegade::new();
    for (p, y) in train {
        model.add(p.clone(), *y);
    }
    let _ = model.predict(&test[0].0); // force training
    let diag = model.diagnostics();

    let mut sse_predict = 0.0;
    let mut sse_k5 = 0.0;
    for (p, y) in test {
        let pred = model.predict(p);
        sse_predict += (pred - y).powi(2);
        let pred_k5 = model.predict_k(p, 5);
        sse_k5 += (pred_k5 - y).powi(2);
    }
    let rmse_predict = (sse_predict / test.len() as f64).sqrt();
    let rmse_k5 = (sse_k5 / test.len() as f64).sqrt();

    eprintln!("=== Wine Quality: Gaussian Kernel vs Hard K ===");
    eprintln!("  n_train={}, n_test={}", train.len(), test.len());
    eprintln!("  Mean={:.2}, std={:.2}", mean_y, std_y);
    eprintln!(
        "  Auto k={}, bandwidth={:?}, metric={}",
        diag.optimal_k.unwrap_or(0),
        diag.kernel_bandwidth,
        diag.metric_active
    );
    eprintln!("  predict() RMSE:   {:.4}", rmse_predict);
    eprintln!("  predict_k(5) RMSE: {:.4}", rmse_k5);
    eprintln!("  Improvement: {:.4}", rmse_k5 - rmse_predict);
}
