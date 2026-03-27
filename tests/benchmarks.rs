use renegade_ml::{DataPoint, Renegade};

// --- Iris dataset ---

#[derive(Clone, Debug)]
struct IrisPoint {
    features: [f64; 4],
    ranges: [(f64, f64); 4],
}

impl DataPoint for IrisPoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        (0..4)
            .map(|i| {
                (self.features[i] - other.features[i]).abs() / (self.ranges[i].1 - self.ranges[i].0)
            })
            .collect()
    }

    fn feature_values(&self) -> Vec<f64> {
        self.features.to_vec()
    }
}

fn load_iris() -> Vec<(IrisPoint, f64)> {
    let data = include_str!("../testdata/iris.csv");
    let rows: Vec<Vec<f64>> = data
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    // Compute ranges
    let mut mins = [f64::MAX; 4];
    let mut maxs = [f64::MIN; 4];
    for row in &rows {
        for i in 0..4 {
            mins[i] = mins[i].min(row[i]);
            maxs[i] = maxs[i].max(row[i]);
        }
    }
    let ranges = std::array::from_fn(|i| (mins[i], maxs[i]));

    rows.into_iter()
        .map(|row| {
            let point = IrisPoint {
                features: [row[0], row[1], row[2], row[3]],
                ranges,
            };
            (point, row[4])
        })
        .collect()
}

// --- Wine dataset ---

#[derive(Clone, Debug)]
struct WinePoint {
    features: [f64; 13],
    ranges: [(f64, f64); 13],
}

impl DataPoint for WinePoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        (0..13)
            .map(|i| {
                let range = self.ranges[i].1 - self.ranges[i].0;
                if range == 0.0 {
                    0.0
                } else {
                    (self.features[i] - other.features[i]).abs() / range
                }
            })
            .collect()
    }

    fn feature_values(&self) -> Vec<f64> {
        self.features.to_vec()
    }
}

fn load_wine() -> Vec<(WinePoint, f64)> {
    let data = include_str!("../testdata/wine.csv");
    let rows: Vec<Vec<f64>> = data
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let mut mins = [f64::MAX; 13];
    let mut maxs = [f64::MIN; 13];
    for row in &rows {
        for i in 0..13 {
            mins[i] = mins[i].min(row[i + 1]);
            maxs[i] = maxs[i].max(row[i + 1]);
        }
    }
    let ranges = std::array::from_fn(|i| (mins[i], maxs[i]));

    rows.into_iter()
        .map(|row| {
            let mut features = [0.0; 13];
            for i in 0..13 {
                features[i] = row[i + 1];
            }
            let point = WinePoint { features, ranges };
            (point, row[0])
        })
        .collect()
}

// --- Auto MPG dataset (mixed types: numeric + categorical origin) ---

#[derive(Clone, Debug)]
struct AutoMpgPoint {
    cylinders: f64,
    displacement: f64,
    horsepower: f64,
    weight: f64,
    acceleration: f64,
    model_year: f64,
    origin: u8, // 1=USA, 2=Europe, 3=Japan (categorical)
    ranges: AutoMpgRanges,
}

#[derive(Clone, Debug)]
struct AutoMpgRanges {
    cylinders: (f64, f64),
    displacement: (f64, f64),
    horsepower: (f64, f64),
    weight: (f64, f64),
    acceleration: (f64, f64),
    model_year: (f64, f64),
}

impl DataPoint for AutoMpgPoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        let r = &self.ranges;
        vec![
            (self.cylinders - other.cylinders).abs() / (r.cylinders.1 - r.cylinders.0),
            (self.displacement - other.displacement).abs() / (r.displacement.1 - r.displacement.0),
            (self.horsepower - other.horsepower).abs() / (r.horsepower.1 - r.horsepower.0),
            (self.weight - other.weight).abs() / (r.weight.1 - r.weight.0),
            (self.acceleration - other.acceleration).abs() / (r.acceleration.1 - r.acceleration.0),
            (self.model_year - other.model_year).abs() / (r.model_year.1 - r.model_year.0),
            if self.origin == other.origin {
                0.0
            } else {
                1.0
            },
        ]
    }

    fn feature_values(&self) -> Vec<f64> {
        vec![
            self.cylinders,
            self.displacement,
            self.horsepower,
            self.weight,
            self.acceleration,
            self.model_year,
            self.origin as f64,
        ]
    }
}

fn load_auto_mpg() -> Vec<(AutoMpgPoint, f64)> {
    let data = include_str!("../testdata/auto_mpg.csv");
    let rows: Vec<Vec<f64>> = data
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    // Compute ranges for numeric features
    let mut cyl_range = (f64::MAX, f64::MIN);
    let mut disp_range = (f64::MAX, f64::MIN);
    let mut hp_range = (f64::MAX, f64::MIN);
    let mut wt_range = (f64::MAX, f64::MIN);
    let mut acc_range = (f64::MAX, f64::MIN);
    let mut yr_range = (f64::MAX, f64::MIN);

    for row in &rows {
        cyl_range.0 = cyl_range.0.min(row[1]);
        cyl_range.1 = cyl_range.1.max(row[1]);
        disp_range.0 = disp_range.0.min(row[2]);
        disp_range.1 = disp_range.1.max(row[2]);
        hp_range.0 = hp_range.0.min(row[3]);
        hp_range.1 = hp_range.1.max(row[3]);
        wt_range.0 = wt_range.0.min(row[4]);
        wt_range.1 = wt_range.1.max(row[4]);
        acc_range.0 = acc_range.0.min(row[5]);
        acc_range.1 = acc_range.1.max(row[5]);
        yr_range.0 = yr_range.0.min(row[6]);
        yr_range.1 = yr_range.1.max(row[6]);
    }

    let ranges = AutoMpgRanges {
        cylinders: cyl_range,
        displacement: disp_range,
        horsepower: hp_range,
        weight: wt_range,
        acceleration: acc_range,
        model_year: yr_range,
    };

    rows.into_iter()
        .map(|row| {
            let point = AutoMpgPoint {
                cylinders: row[1],
                displacement: row[2],
                horsepower: row[3],
                weight: row[4],
                acceleration: row[5],
                model_year: row[6],
                origin: row[7] as u8,
                ranges: ranges.clone(),
            };
            (point, row[0]) // mpg is the target
        })
        .collect()
}

/// Leave-one-out cross-validation for classification.
/// Returns accuracy (fraction correct).
fn loo_classification_accuracy<P: DataPoint + Clone>(data: &[(P, f64)], k: usize) -> f64 {
    let n = data.len();
    let mut correct = 0;

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (point, output)) in data.iter().enumerate() {
            if j != i {
                model.add(point.clone(), *output);
            }
        }

        let neighbors = model.query_k(&data[i].0, k);
        let votes = neighbors.class_votes();
        let predicted_class = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        if (predicted_class - data[i].1).abs() < 0.5 {
            correct += 1;
        }
    }

    correct as f64 / n as f64
}

/// Leave-one-out cross-validation for regression.
/// Returns RMSE.
fn loo_regression_rmse<P: DataPoint + Clone>(data: &[(P, f64)], k: usize) -> f64 {
    let n = data.len();
    let mut sum_sq_err = 0.0;

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (point, output)) in data.iter().enumerate() {
            if j != i {
                model.add(point.clone(), *output);
            }
        }

        let neighbors = model.query_k(&data[i].0, k);
        let predicted = neighbors.weighted_mean();
        let err = predicted - data[i].1;
        sum_sq_err += err * err;
    }

    (sum_sq_err / n as f64).sqrt()
}

#[test]
fn iris_classification_accuracy() {
    let data = load_iris();
    let accuracy = loo_classification_accuracy(&data, 5);
    eprintln!("Iris LOO accuracy (k=5): {:.1}%", accuracy * 100.0);
    assert!(
        accuracy >= 0.93,
        "Iris accuracy {:.1}% below 93% threshold",
        accuracy * 100.0
    );
}

#[test]
fn wine_classification_accuracy() {
    let data = load_wine();
    let accuracy = loo_classification_accuracy(&data, 5);
    eprintln!("Wine LOO accuracy (k=5): {:.1}%", accuracy * 100.0);
    assert!(
        accuracy >= 0.93,
        "Wine accuracy {:.1}% below 93% threshold",
        accuracy * 100.0
    );
}

#[test]
fn auto_mpg_regression_rmse() {
    let data = load_auto_mpg();
    let rmse = loo_regression_rmse(&data, 5);

    // Also compute mean for context
    let mean_mpg: f64 = data.iter().map(|(_, mpg)| mpg).sum::<f64>() / data.len() as f64;
    eprintln!(
        "Auto MPG LOO RMSE (k=5): {:.2} (mean MPG: {:.1})",
        rmse, mean_mpg
    );
    assert!(rmse < 4.5, "Auto MPG RMSE {:.2} above 4.5 threshold", rmse);
}

#[test]
fn iris_extrapolation_sanity() {
    let data = load_iris();

    // Build model with all data
    let mut model = Renegade::new();
    for (point, class) in &data {
        model.add(point.clone(), *class);
    }

    // Predict for a point clearly in the setosa region
    let setosa_query = IrisPoint {
        features: [5.0, 3.5, 1.4, 0.2],
        ranges: data[0].0.ranges,
    };
    let pred = model.predict_k_extrapolated(&setosa_query, 10);
    eprintln!(
        "Setosa query: predicted={:.2}, r²={:.3}",
        pred.value, pred.r_squared
    );
    // Should predict near class 0
    assert!(
        pred.value < 0.5,
        "Expected prediction near 0 (setosa), got {:.2}",
        pred.value
    );
}

// --- Auto-K benchmarks ---

/// LOO accuracy using auto-selected K.
fn loo_auto_classification_accuracy<P: DataPoint + Clone>(data: &[(P, f64)]) -> (f64, usize) {
    let n = data.len();
    let mut correct = 0;

    // Compute optimal K on full dataset (this is what a user would do).
    let mut full_model = Renegade::new();
    for (point, output) in data {
        full_model.add(point.clone(), *output);
    }
    let k = full_model.get_optimal_k();

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (point, output)) in data.iter().enumerate() {
            if j != i {
                model.add(point.clone(), *output);
            }
        }

        let neighbors = model.query_k(&data[i].0, k);
        let votes = neighbors.class_votes();
        let predicted_class = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        if (predicted_class - data[i].1).abs() < 0.5 {
            correct += 1;
        }
    }

    (correct as f64 / n as f64, k)
}

fn loo_auto_regression_rmse<P: DataPoint + Clone>(data: &[(P, f64)]) -> (f64, usize) {
    let n = data.len();
    let mut sum_sq_err = 0.0;

    let mut full_model = Renegade::new();
    for (point, output) in data {
        full_model.add(point.clone(), *output);
    }
    let k = full_model.get_optimal_k();

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (point, output)) in data.iter().enumerate() {
            if j != i {
                model.add(point.clone(), *output);
            }
        }

        let neighbors = model.query_k(&data[i].0, k);
        let predicted = neighbors.weighted_mean();
        let err = predicted - data[i].1;
        sum_sq_err += err * err;
    }

    ((sum_sq_err / n as f64).sqrt(), k)
}

#[test]
fn iris_auto_k() {
    let data = load_iris();
    let (accuracy, k) = loo_auto_classification_accuracy(&data);
    eprintln!("Iris LOO accuracy (auto k={}): {:.1}%", k, accuracy * 100.0);
    assert!(
        accuracy >= 0.93,
        "Iris auto-K accuracy {:.1}% below 93% threshold",
        accuracy * 100.0
    );
}

#[test]
fn wine_auto_k() {
    let data = load_wine();
    let (accuracy, k) = loo_auto_classification_accuracy(&data);
    eprintln!("Wine LOO accuracy (auto k={}): {:.1}%", k, accuracy * 100.0);
    assert!(
        accuracy >= 0.93,
        "Wine auto-K accuracy {:.1}% below 93% threshold",
        accuracy * 100.0
    );
}

#[test]
fn auto_mpg_auto_k() {
    let data = load_auto_mpg();
    let (rmse, k) = loo_auto_regression_rmse(&data);
    let mean_mpg: f64 = data.iter().map(|(_, mpg)| mpg).sum::<f64>() / data.len() as f64;
    eprintln!(
        "Auto MPG LOO RMSE (auto k={}): {:.2} (mean MPG: {:.1})",
        k, rmse, mean_mpg
    );
    assert!(
        rmse < 4.5,
        "Auto MPG auto-K RMSE {:.2} above 4.5 threshold",
        rmse
    );
}
