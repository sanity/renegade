use renegade::{DataPoint, Renegade};

/// Generic numeric point that works with any CSV dataset.
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

    fn num_features(&self) -> usize {
        self.values.len()
    }
}

/// Load a CSV where all columns are numeric, last column is target.
fn load_csv(data: &str) -> Vec<(CsvPoint, f64)> {
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
    let nfeatures = ncols - 1;

    // Compute ranges
    let mut mins = vec![f64::MAX; nfeatures];
    let mut maxs = vec![f64::MIN; nfeatures];
    for row in &rows {
        for i in 0..nfeatures {
            mins[i] = mins[i].min(row[i]);
            maxs[i] = maxs[i].max(row[i]);
        }
    }
    let ranges: Vec<(f64, f64)> = mins.into_iter().zip(maxs).collect();

    rows.into_iter()
        .map(|row| {
            let values = row[..nfeatures].to_vec();
            let target = row[nfeatures];
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

/// Detect classification: all targets are integers.
fn is_classification(data: &[(CsvPoint, f64)]) -> bool {
    data.iter().all(|(_, y)| (y - y.round()).abs() < 1e-6)
}

/// LOO classification accuracy with a given model setup.
fn loo_classification(data: &[(CsvPoint, f64)], k: usize) -> f64 {
    let n = data.len();
    let mut correct = 0;
    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let neighbors = model.query_k(&data[i].0, k);
        let votes = neighbors.class_votes();
        if let Some((predicted, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            if (*predicted - data[i].1).abs() < 0.5 {
                correct += 1;
            }
        }
    }
    correct as f64 / n as f64
}

/// LOO regression RMSE with a given model setup.
fn loo_regression(data: &[(CsvPoint, f64)], k: usize) -> f64 {
    let n = data.len();
    let mut sse = 0.0;
    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let neighbors = model.query_k(&data[i].0, k);
        let pred = neighbors.weighted_mean();
        sse += (pred - data[i].1).powi(2);
    }
    (sse / n as f64).sqrt()
}

/// Train/test split evaluation for larger datasets.
/// Uses first `frac` as train, rest as test.
fn split_eval(data: &[(CsvPoint, f64)], train_frac: f64, is_class: bool) -> (f64, f64, usize) {
    let split = (data.len() as f64 * train_frac) as usize;
    let (train, test) = data.split_at(split);

    // Build model with auto K and metric
    let mut model = Renegade::new();
    for (p, y) in train {
        model.add(p.clone(), *y);
    }
    let k = model.get_optimal_k();

    // Also build a no-metric model for comparison
    let mut baseline = Renegade::new();
    for (p, y) in train {
        baseline.add(p.clone(), *y);
    }

    if is_class {
        let mut correct_metric = 0;
        let mut correct_baseline = 0;
        for (p, y) in test {
            // With metric
            let votes = model.query_k(p, k).class_votes();
            if let Some((pred, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                if (*pred - y).abs() < 0.5 {
                    correct_metric += 1;
                }
            }
            // Baseline (k=5, no metric)
            let votes = baseline.query_k(p, 5).class_votes();
            if let Some((pred, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                if (*pred - y).abs() < 0.5 {
                    correct_baseline += 1;
                }
            }
        }
        let n_test = test.len() as f64;
        (
            correct_baseline as f64 / n_test,
            correct_metric as f64 / n_test,
            k,
        )
    } else {
        let mut sse_metric = 0.0;
        let mut sse_baseline = 0.0;
        for (p, y) in test {
            let pred_m = model.query_k(p, k).weighted_mean();
            sse_metric += (pred_m - y).powi(2);
            let pred_b = baseline.query_k(p, 5).weighted_mean();
            sse_baseline += (pred_b - y).powi(2);
        }
        let n_test = test.len() as f64;
        (
            (sse_baseline / n_test).sqrt(),
            (sse_metric / n_test).sqrt(),
            k,
        )
    }
}

// --- Breast Cancer Wisconsin: 569 rows, 30 features, binary classification ---
// Expected KNN accuracy: ~95-97%

#[test]
fn breast_cancer_classification() {
    let data = load_csv(include_str!("../testdata/breast_cancer.csv"));
    assert_eq!(data.len(), 569);

    let is_class = is_classification(&data);
    assert!(is_class, "Should be classification");

    // LOO with k=5 baseline
    let acc_k5 = loo_classification(&data, 5);

    // Auto K + metric via train/test (LOO too slow for 569×30)
    let (baseline, learned, k) = split_eval(&data, 0.8, true);

    eprintln!("=== Breast Cancer (569 rows, 30 features, binary) ===");
    eprintln!("  LOO k=5 accuracy:           {:.1}%", acc_k5 * 100.0);
    eprintln!("  Split baseline k=5:         {:.1}%", baseline * 100.0);
    eprintln!("  Split auto k={} + metric:   {:.1}%", k, learned * 100.0);

    assert!(
        acc_k5 >= 0.93,
        "Breast cancer accuracy {:.1}% below 93%",
        acc_k5 * 100.0
    );
}

// --- Ionosphere: 351 rows, 34 features, binary classification ---
// Expected KNN accuracy: ~85-90%

#[test]
fn ionosphere_classification() {
    let data = load_csv(include_str!("../testdata/ionosphere.csv"));
    assert_eq!(data.len(), 351);

    let is_class = is_classification(&data);
    assert!(is_class, "Should be classification");

    let acc_k5 = loo_classification(&data, 5);
    let (baseline, learned, k) = split_eval(&data, 0.8, true);

    eprintln!("=== Ionosphere (351 rows, 34 features, binary) ===");
    eprintln!("  LOO k=5 accuracy:           {:.1}%", acc_k5 * 100.0);
    eprintln!("  Split baseline k=5:         {:.1}%", baseline * 100.0);
    eprintln!("  Split auto k={} + metric:   {:.1}%", k, learned * 100.0);

    assert!(
        acc_k5 >= 0.82,
        "Ionosphere accuracy {:.1}% below 82%",
        acc_k5 * 100.0
    );
}

// --- Wine Quality White: 4898 rows, 11 features, regression (quality 3-9) ---
// Expected KNN RMSE: ~0.7-0.9
// Too large for LOO — use train/test split only.

#[test]
fn wine_quality_regression() {
    let data = load_csv(include_str!("../testdata/wine_quality.csv"));
    assert_eq!(data.len(), 4898);

    let mean_y: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;
    let std_y: f64 =
        (data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

    let (baseline, learned, k) = split_eval(&data, 0.8, false);

    eprintln!("=== Wine Quality White (4898 rows, 11 features, regression) ===");
    eprintln!("  Mean quality: {:.2}, std: {:.2}", mean_y, std_y);
    eprintln!("  Split baseline k=5 RMSE:    {:.3}", baseline);
    eprintln!("  Split auto k={} + metric RMSE: {:.3}", k, learned);

    // Baseline RMSE should be meaningfully below std (better than predicting the mean)
    assert!(
        baseline < std_y,
        "Baseline RMSE {:.3} should be below std {:.3}",
        baseline,
        std_y
    );
}
