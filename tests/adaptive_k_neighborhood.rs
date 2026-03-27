//! Experiment 2: Predict optimal k from neighborhood features.
//!
//! Instead of predicting k from the query's input features (which failed),
//! we query at max_k, then extract features from the neighbor set itself:
//!   - Distance profile (nearest, median, max, gaps)
//!   - Output profile (variance, entropy, agreement)
//!   - Local density
//!
//! These are properties of the local neighborhood that should correlate with
//! how many neighbors are "trustworthy" for this specific query.

use renegade_ml::{DataPoint, Neighbor, Renegade};

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

/// Extract features from a set of max_k neighbors that describe the local
/// neighborhood structure. These features should correlate with optimal k.
#[derive(Clone, Debug)]
struct NeighborhoodFeatures {
    values: Vec<f64>,
    ranges: Vec<(f64, f64)>,
}

impl DataPoint for NeighborhoodFeatures {
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

const NUM_NEIGHBORHOOD_FEATURES: usize = 10;

fn extract_neighborhood_features(neighbors: &[Neighbor], max_k: usize) -> Vec<f64> {
    let k = neighbors.len().min(max_k);
    if k == 0 {
        return vec![0.0; NUM_NEIGHBORHOOD_FEATURES];
    }

    let dists: Vec<f64> = neighbors[..k].iter().map(|n| n.distance).collect();
    let outputs: Vec<f64> = neighbors[..k].iter().map(|n| n.output).collect();

    // Distance features
    let d_min = dists[0];
    let d_max = dists[k - 1];
    let d_mean = dists.iter().sum::<f64>() / k as f64;
    let d_median = dists[k / 2];

    // Distance ratio: how spread out are the neighbors?
    let d_ratio = if d_min > 0.0 {
        d_max / d_min
    } else {
        d_max * 100.0 + 1.0
    };

    // Max gap in distances (normalized by d_max): where's the natural "cutoff"?
    let max_gap = if k > 1 && d_max > 0.0 {
        (1..k)
            .map(|i| dists[i] - dists[i - 1])
            .fold(0.0f64, f64::max)
            / d_max
    } else {
        0.0
    };

    // Index of max gap (normalized to [0, 1]): where in the neighbor list is the cutoff?
    let max_gap_pos = if k > 1 {
        let pos = (1..k)
            .max_by(|&i, &j| {
                let gi = dists[i] - dists[i - 1];
                let gj = dists[j] - dists[j - 1];
                gi.partial_cmp(&gj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(1);
        pos as f64 / k as f64
    } else {
        0.0
    };

    // Output features
    let o_mean = outputs.iter().sum::<f64>() / k as f64;
    let o_var = outputs.iter().map(|&o| (o - o_mean).powi(2)).sum::<f64>() / k as f64;

    // Output agreement among nearest few vs all: do close neighbors agree more?
    let half_k = (k / 2).max(1);
    let o_mean_near = outputs[..half_k].iter().sum::<f64>() / half_k as f64;
    let o_mean_far = if k > half_k {
        outputs[half_k..].iter().sum::<f64>() / (k - half_k) as f64
    } else {
        o_mean_near
    };
    let near_far_diff = (o_mean_near - o_mean_far).abs();

    vec![
        d_min,         // 0: distance to nearest
        d_mean,        // 1: mean distance
        d_median,      // 2: median distance
        d_ratio,       // 3: spread of neighbor distances
        max_gap,       // 4: largest gap (normalized)
        max_gap_pos,   // 5: position of largest gap
        o_var,         // 6: output variance among neighbors
        near_far_diff, // 7: near vs far output disagreement
        d_max,         // 8: distance to farthest neighbor
        k as f64,      // 9: actual number of neighbors returned
    ]
}

struct NeighborhoodAnalysis {
    global_k: usize,
    global_error: f64,
    oracle_error: f64,
    neighborhood_predicted_error: f64,
    /// Simple heuristic: use k = position of max distance gap
    gap_heuristic_error: f64,
    /// Heuristic: use k where output variance first stabilizes
    variance_heuristic_error: f64,
}

fn analyze_neighborhood_k(data: &[(CsvPoint, f64)]) -> NeighborhoodAnalysis {
    let n = data.len();
    let max_k = ((n - 1) as f64).sqrt().ceil() as usize;
    let max_k = max_k.max(1).min(n - 1);
    let is_class = is_classification(data);

    // Phase 1: Compute per-point errors and best-k, plus neighborhood features
    let mut best_k_per_point = Vec::with_capacity(n);
    let mut errors_per_point: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut nhood_features: Vec<Vec<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }

        let neighbors = model.query_k(&data[i].0, max_k);
        let nb = &neighbors.neighbors;

        // Extract neighborhood features
        nhood_features.push(extract_neighborhood_features(nb, max_k));

        // Compute error at each k
        let mut errors_for_k = vec![0.0f64; max_k + 1];

        if is_class {
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

        let min_error = errors_for_k[1..=max_k]
            .iter()
            .copied()
            .fold(f64::MAX, f64::min);
        let epsilon = if is_class { 0.0 } else { min_error * 0.05 };
        let best_k = (1..=max_k)
            .find(|&k| errors_for_k[k] <= min_error + epsilon)
            .unwrap_or(1);

        best_k_per_point.push(best_k);
        errors_per_point.push(errors_for_k);
    }

    // Global optimal k
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
    let global_error = global_errors[global_k] / n as f64;

    let oracle_error: f64 = (0..n)
        .map(|i| errors_per_point[i][best_k_per_point[i]])
        .sum::<f64>()
        / n as f64;

    // Phase 2: Neighborhood-features meta-model via LOO
    // Compute ranges for neighborhood features
    let mut nf_mins = vec![f64::MAX; NUM_NEIGHBORHOOD_FEATURES];
    let mut nf_maxs = vec![f64::MIN; NUM_NEIGHBORHOOD_FEATURES];
    for feats in &nhood_features {
        for (fi, &v) in feats.iter().enumerate() {
            nf_mins[fi] = nf_mins[fi].min(v);
            nf_maxs[fi] = nf_maxs[fi].max(v);
        }
    }
    let nf_ranges: Vec<(f64, f64)> = nf_mins.into_iter().zip(nf_maxs).collect();

    let mut predicted_error_total = 0.0;
    for i in 0..n {
        let mut k_model: Renegade<NeighborhoodFeatures> = Renegade::new();
        for j in 0..n {
            if j != i {
                let nf = NeighborhoodFeatures {
                    values: nhood_features[j].clone(),
                    ranges: nf_ranges.clone(),
                };
                k_model.add(nf, best_k_per_point[j] as f64);
            }
        }
        let query_nf = NeighborhoodFeatures {
            values: nhood_features[i].clone(),
            ranges: nf_ranges.clone(),
        };
        let pred_k_raw = k_model.predict(&query_nf);
        let pred_k = pred_k_raw.round().max(1.0).min(max_k as f64) as usize;
        predicted_error_total += errors_per_point[i][pred_k];
    }
    let neighborhood_predicted_error = predicted_error_total / n as f64;

    // Phase 3: Simple heuristics (no meta-model needed)

    // Gap heuristic: use k = position of largest distance gap
    let mut gap_error_total = 0.0;
    for i in 0..n {
        let feats = &nhood_features[i];
        // max_gap_pos is feats[5], normalized to [0,1]
        let gap_k = (feats[5] * max_k as f64).round().max(1.0).min(max_k as f64) as usize;
        gap_error_total += errors_per_point[i][gap_k];
    }
    let gap_heuristic_error = gap_error_total / n as f64;

    // Variance heuristic: for each point, find smallest k where adding the next
    // neighbor would increase output variance significantly
    let mut var_error_total = 0.0;
    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let neighbors = model.query_k(&data[i].0, max_k);
        let nb = &neighbors.neighbors;

        // Find k where output variance jumps
        let mut best_var_k = max_k;
        if nb.len() >= 3 {
            let mut outputs_so_far: Vec<f64> = Vec::new();
            let mut prev_var = 0.0;
            for k in 1..=max_k.min(nb.len()) {
                outputs_so_far.push(nb[k - 1].output);
                if k >= 2 {
                    let mean = outputs_so_far.iter().sum::<f64>() / k as f64;
                    let var = outputs_so_far
                        .iter()
                        .map(|&o| (o - mean).powi(2))
                        .sum::<f64>()
                        / k as f64;
                    // If variance jumps by >50% from adding this neighbor, stop at k-1
                    if k >= 3 && prev_var > 0.0 && var > prev_var * 1.5 {
                        best_var_k = k - 1;
                        break;
                    }
                    prev_var = var;
                }
            }
        } else {
            best_var_k = nb.len().max(1);
        }
        best_var_k = best_var_k.max(1).min(max_k);
        var_error_total += errors_per_point[i][best_var_k];
    }
    let variance_heuristic_error = var_error_total / n as f64;

    NeighborhoodAnalysis {
        global_k,
        global_error,
        oracle_error,
        neighborhood_predicted_error,
        gap_heuristic_error,
        variance_heuristic_error,
    }
}

fn print_neighborhood_analysis(name: &str, data: &[(CsvPoint, f64)], a: &NeighborhoodAnalysis) {
    let is_class = is_classification(data);
    let fmt = |e: f64| -> String {
        if is_class {
            format!("{:.1}% error", e * 100.0)
        } else {
            format!("{:.4} RMSE", e.sqrt())
        }
    };

    eprintln!("\n=== {} (n={}) ===", name, data.len());
    eprintln!("  Global k={}: {}", a.global_k, fmt(a.global_error));
    eprintln!("  Oracle (per-point best): {}", fmt(a.oracle_error));
    eprintln!(
        "  Neighborhood meta-model: {}",
        fmt(a.neighborhood_predicted_error)
    );
    eprintln!("  Gap heuristic: {}", fmt(a.gap_heuristic_error));
    eprintln!("  Variance heuristic: {}", fmt(a.variance_heuristic_error));

    let headroom = if is_class {
        (a.global_error - a.oracle_error) * 100.0
    } else {
        a.global_error.sqrt() - a.oracle_error.sqrt()
    };
    let nhood_gain = if is_class {
        (a.global_error - a.neighborhood_predicted_error) * 100.0
    } else {
        a.global_error.sqrt() - a.neighborhood_predicted_error.sqrt()
    };
    let gap_gain = if is_class {
        (a.global_error - a.gap_heuristic_error) * 100.0
    } else {
        a.global_error.sqrt() - a.gap_heuristic_error.sqrt()
    };
    let var_gain = if is_class {
        (a.global_error - a.variance_heuristic_error) * 100.0
    } else {
        a.global_error.sqrt() - a.variance_heuristic_error.sqrt()
    };

    eprintln!("  ---");
    eprintln!(
        "  Headroom: {}{:.4}",
        if headroom >= 0.0 { "+" } else { "" },
        headroom
    );
    eprintln!(
        "  Neighborhood model: {}{:.4}",
        if nhood_gain >= 0.0 { "+" } else { "" },
        nhood_gain
    );
    eprintln!(
        "  Gap heuristic: {}{:.4}",
        if gap_gain >= 0.0 { "+" } else { "" },
        gap_gain
    );
    eprintln!(
        "  Variance heuristic: {}{:.4}",
        if var_gain >= 0.0 { "+" } else { "" },
        var_gain
    );
}

// Only run small/medium datasets — breast cancer LOO is too slow with meta-model
#[test]
fn neighborhood_k_iris() {
    let data = load_csv(include_str!("../testdata/iris.csv"));
    let a = analyze_neighborhood_k(&data);
    print_neighborhood_analysis("Iris", &data, &a);
}

#[test]
fn neighborhood_k_wine() {
    let data = load_csv_target(include_str!("../testdata/wine.csv"), Some(0));
    let a = analyze_neighborhood_k(&data);
    print_neighborhood_analysis("Wine", &data, &a);
}

#[test]
fn neighborhood_k_auto_mpg() {
    let data = load_csv_target(include_str!("../testdata/auto_mpg.csv"), Some(0));
    let a = analyze_neighborhood_k(&data);
    print_neighborhood_analysis("Auto MPG", &data, &a);
}

#[test]
fn neighborhood_k_ionosphere() {
    let data = load_csv(include_str!("../testdata/ionosphere.csv"));
    let a = analyze_neighborhood_k(&data);
    print_neighborhood_analysis("Ionosphere", &data, &a);
}
