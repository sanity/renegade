//! Experiment 3: Soft weighting with adaptive kernel bandwidth.
//!
//! Instead of discrete k (hard cutoff), use ALL max_k neighbors with a
//! smooth kernel that decays with distance. The bandwidth h controls
//! the effective "softness" — small h ≈ small k, large h ≈ large k.
//!
//! Tested approaches:
//!   1. Baseline: hard k + inverse-distance weighting (current renegade)
//!   2. Gaussian kernel with fixed bandwidth (LOO-tuned)
//!   3. Adaptive bandwidth: h = distance to kth neighbor (varies per query)
//!   4. Dual adaptive: h from local density, separate LOO for the density scale

use renegade_ml::{DataPoint, Renegade};
use std::collections::HashSet;

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
            .collect::<HashSet<_>>()
            .len()
            <= 20
}

/// Neighbor info extracted during LOO.
struct NeighborInfo {
    distance: f64,
    output: f64,
    weight: f64, // instance weight (always 1.0 for this experiment)
}

/// Predict using hard k cutoff + inverse-distance weighting (current renegade behavior).
fn predict_hard_k(neighbors: &[NeighborInfo], k: usize, is_class: bool) -> f64 {
    let nb = &neighbors[..k.min(neighbors.len())];
    if is_class {
        // Weighted class voting
        let mut votes: Vec<(f64, f64)> = Vec::new();
        for n in nb {
            let w = if n.distance == 0.0 {
                1e12
            } else {
                n.weight / n.distance
            };
            if let Some(entry) = votes
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += w;
            } else {
                votes.push((n.output, w));
            }
        }
        votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|v| v.0)
            .unwrap_or(0.0)
    } else {
        // Inverse-distance weighted mean
        let mut ws = 0.0;
        let mut vs = 0.0;
        let mut exact = None;
        for n in nb {
            if n.distance == 0.0 {
                exact = Some(n.output);
                break;
            }
            let w = n.weight / n.distance;
            ws += w;
            vs += w * n.output;
        }
        exact.unwrap_or_else(|| if ws > 0.0 { vs / ws } else { f64::NAN })
    }
}

/// Predict using Gaussian kernel: w(d) = exp(-d²/(2h²)).
/// Uses ALL max_k neighbors, weighted by kernel.
fn predict_gaussian(neighbors: &[NeighborInfo], h: f64, is_class: bool) -> f64 {
    if h <= 0.0 {
        // Degenerate: just use nearest neighbor
        return neighbors.first().map(|n| n.output).unwrap_or(f64::NAN);
    }

    let h2 = 2.0 * h * h;

    if is_class {
        let mut votes: Vec<(f64, f64)> = Vec::new();
        for n in neighbors {
            let w = (-n.distance * n.distance / h2).exp() * n.weight;
            if w < 1e-12 {
                continue;
            }
            if let Some(entry) = votes
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += w;
            } else {
                votes.push((n.output, w));
            }
        }
        votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|v| v.0)
            .unwrap_or(neighbors.first().map(|n| n.output).unwrap_or(0.0))
    } else {
        let mut ws = 0.0;
        let mut vs = 0.0;
        for n in neighbors {
            let w = (-n.distance * n.distance / h2).exp() * n.weight;
            if w < 1e-12 {
                continue;
            }
            ws += w;
            vs += w * n.output;
        }
        if ws > 0.0 {
            vs / ws
        } else {
            neighbors.first().map(|n| n.output).unwrap_or(f64::NAN)
        }
    }
}

/// Predict using tricube kernel: w(d) = (1 - (d/h)³)³ for d < h, else 0.
/// Compact support = natural cutoff, smoother than hard k.
fn predict_tricube(neighbors: &[NeighborInfo], h: f64, is_class: bool) -> f64 {
    if h <= 0.0 {
        return neighbors.first().map(|n| n.output).unwrap_or(f64::NAN);
    }

    if is_class {
        let mut votes: Vec<(f64, f64)> = Vec::new();
        for n in neighbors {
            let u = n.distance / h;
            if u >= 1.0 {
                continue;
            }
            let w = (1.0 - u * u * u).powi(3) * n.weight;
            if let Some(entry) = votes
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += w;
            } else {
                votes.push((n.output, w));
            }
        }
        votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|v| v.0)
            .unwrap_or(neighbors.first().map(|n| n.output).unwrap_or(0.0))
    } else {
        let mut ws = 0.0;
        let mut vs = 0.0;
        for n in neighbors {
            let u = n.distance / h;
            if u >= 1.0 {
                continue;
            }
            let w = (1.0 - u * u * u).powi(3) * n.weight;
            ws += w;
            vs += w * n.output;
        }
        if ws > 0.0 {
            vs / ws
        } else {
            neighbors.first().map(|n| n.output).unwrap_or(f64::NAN)
        }
    }
}

fn compute_error(predicted: f64, actual: f64, is_class: bool) -> f64 {
    if is_class {
        if (predicted - actual).abs() > 0.5 {
            1.0
        } else {
            0.0
        }
    } else {
        (predicted - actual).powi(2)
    }
}

struct SoftWeightingAnalysis {
    n: usize,
    is_class: bool,
    max_k: usize,
    // Baseline: current renegade (hard k, inverse distance)
    baseline_k: usize,
    baseline_error: f64,
    // Fixed Gaussian kernel (LOO-optimal h)
    gaussian_fixed_h: f64,
    gaussian_fixed_error: f64,
    // Adaptive Gaussian: h = distance to k_ref'th neighbor (LOO-optimal k_ref)
    gaussian_adaptive_k_ref: usize,
    gaussian_adaptive_error: f64,
    // Fixed tricube kernel
    tricube_fixed_h: f64,
    tricube_fixed_error: f64,
    // Adaptive tricube: h = distance to k_ref'th neighbor
    tricube_adaptive_k_ref: usize,
    tricube_adaptive_error: f64,
    // Oracle: per-point best bandwidth (upper bound)
    oracle_error: f64,
}

fn analyze_soft_weighting(data: &[(CsvPoint, f64)]) -> SoftWeightingAnalysis {
    let n = data.len();
    let max_k = ((n - 1) as f64).sqrt().ceil() as usize;
    let max_k = max_k.max(1).min(n - 1);
    let is_class = is_classification(data);

    // Phase 1: For each LOO point, get max_k neighbors and their info
    let mut all_neighbors: Vec<Vec<NeighborInfo>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut model = Renegade::new();
        for (j, (p, y)) in data.iter().enumerate() {
            if j != i {
                model.add(p.clone(), *y);
            }
        }
        let result = model.query_k(&data[i].0, max_k);
        let nb: Vec<NeighborInfo> = result
            .neighbors
            .iter()
            .map(|neighbor| NeighborInfo {
                distance: neighbor.distance,
                output: neighbor.output,
                weight: neighbor.weight,
            })
            .collect();
        all_neighbors.push(nb);
    }

    // Phase 2: Evaluate baseline (hard k + inverse distance) for each k
    let mut baseline_errors_by_k = vec![0.0f64; max_k + 1];
    for (i, nb) in all_neighbors.iter().enumerate() {
        for k in 1..=max_k {
            let pred = predict_hard_k(nb, k, is_class);
            baseline_errors_by_k[k] += compute_error(pred, data[i].1, is_class);
        }
    }
    let baseline_k = (1..=max_k)
        .min_by(|&a, &b| {
            baseline_errors_by_k[a]
                .partial_cmp(&baseline_errors_by_k[b])
                .unwrap()
        })
        .unwrap_or(1);
    let baseline_error = baseline_errors_by_k[baseline_k] / n as f64;

    // Phase 3: Fixed Gaussian kernel — sweep h values
    // Use percentiles of the overall distance distribution as candidates
    let mut all_dists: Vec<f64> = Vec::new();
    for nb in &all_neighbors {
        for n_info in nb {
            if n_info.distance > 0.0 {
                all_dists.push(n_info.distance);
            }
        }
    }
    all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Try 20 bandwidth values spanning the distance distribution
    let h_candidates: Vec<f64> = (1..=20)
        .map(|i| {
            let pct = i as f64 / 21.0;
            let idx = (pct * all_dists.len() as f64) as usize;
            all_dists[idx.min(all_dists.len() - 1)]
        })
        .collect();

    let mut best_gauss_h = h_candidates[0];
    let mut best_gauss_error = f64::MAX;
    let mut best_tri_h = h_candidates[0];
    let mut best_tri_error = f64::MAX;

    for &h in &h_candidates {
        let mut gauss_err = 0.0;
        let mut tri_err = 0.0;
        for (i, nb) in all_neighbors.iter().enumerate() {
            gauss_err += compute_error(predict_gaussian(nb, h, is_class), data[i].1, is_class);
            tri_err += compute_error(predict_tricube(nb, h, is_class), data[i].1, is_class);
        }
        if gauss_err < best_gauss_error {
            best_gauss_error = gauss_err;
            best_gauss_h = h;
        }
        if tri_err < best_tri_error {
            best_tri_error = tri_err;
            best_tri_h = h;
        }
    }
    let gaussian_fixed_error = best_gauss_error / n as f64;
    let tricube_fixed_error = best_tri_error / n as f64;

    // Phase 4: Adaptive kernel — h = distance to k_ref'th neighbor (per query)
    // Sweep k_ref to find the best reference neighbor index
    let mut best_gauss_adaptive_kref = 1;
    let mut best_gauss_adaptive_error = f64::MAX;
    let mut best_tri_adaptive_kref = 1;
    let mut best_tri_adaptive_error = f64::MAX;

    for k_ref in 1..=max_k {
        let mut gauss_err = 0.0;
        let mut tri_err = 0.0;
        for (i, nb) in all_neighbors.iter().enumerate() {
            if nb.len() < k_ref {
                continue;
            }
            // h = distance to k_ref'th neighbor (adaptive per query)
            let h = nb[k_ref - 1].distance;
            let h = if h == 0.0 {
                // All k_ref nearest have distance 0; use a small positive h
                // to avoid degenerate kernel
                nb.iter()
                    .find(|n| n.distance > 0.0)
                    .map(|n| n.distance * 0.1)
                    .unwrap_or(0.001)
            } else {
                h
            };

            gauss_err += compute_error(predict_gaussian(nb, h, is_class), data[i].1, is_class);
            tri_err += compute_error(predict_tricube(nb, h, is_class), data[i].1, is_class);
        }
        if gauss_err < best_gauss_adaptive_error {
            best_gauss_adaptive_error = gauss_err;
            best_gauss_adaptive_kref = k_ref;
        }
        if tri_err < best_tri_adaptive_error {
            best_tri_adaptive_error = tri_err;
            best_tri_adaptive_kref = k_ref;
        }
    }
    let gaussian_adaptive_error = best_gauss_adaptive_error / n as f64;
    let tricube_adaptive_error = best_tri_adaptive_error / n as f64;

    // Phase 5: Oracle — per-point best h (from h_candidates + adaptive h values)
    let mut oracle_total = 0.0;
    for (i, nb) in all_neighbors.iter().enumerate() {
        let mut best_err = f64::MAX;

        // Try fixed h values
        for &h in &h_candidates {
            let e = compute_error(predict_gaussian(nb, h, is_class), data[i].1, is_class);
            if e < best_err {
                best_err = e;
            }
            let e = compute_error(predict_tricube(nb, h, is_class), data[i].1, is_class);
            if e < best_err {
                best_err = e;
            }
        }

        // Try adaptive h values (distance to each neighbor)
        for k_ref in 1..=nb.len() {
            let h = if nb[k_ref - 1].distance == 0.0 {
                nb.iter()
                    .find(|n| n.distance > 0.0)
                    .map(|n| n.distance * 0.1)
                    .unwrap_or(0.001)
            } else {
                nb[k_ref - 1].distance
            };
            let e = compute_error(predict_gaussian(nb, h, is_class), data[i].1, is_class);
            if e < best_err {
                best_err = e;
            }
            let e = compute_error(predict_tricube(nb, h, is_class), data[i].1, is_class);
            if e < best_err {
                best_err = e;
            }
        }

        // Also try hard k for oracle
        for k in 1..=nb.len() {
            let e = compute_error(predict_hard_k(nb, k, is_class), data[i].1, is_class);
            if e < best_err {
                best_err = e;
            }
        }

        oracle_total += best_err;
    }
    let oracle_error = oracle_total / n as f64;

    SoftWeightingAnalysis {
        n,
        is_class,
        max_k,
        baseline_k,
        baseline_error,
        gaussian_fixed_h: best_gauss_h,
        gaussian_fixed_error,
        gaussian_adaptive_k_ref: best_gauss_adaptive_kref,
        gaussian_adaptive_error,
        tricube_fixed_h: best_tri_h,
        tricube_fixed_error,
        tricube_adaptive_k_ref: best_tri_adaptive_kref,
        tricube_adaptive_error,
        oracle_error,
    }
}

fn print_soft_analysis(name: &str, a: &SoftWeightingAnalysis) {
    let fmt = |e: f64| -> String {
        if a.is_class {
            format!("{:.2}% error", e * 100.0)
        } else {
            format!("{:.4} RMSE", e.sqrt())
        }
    };
    let gain = |e: f64| -> String {
        let diff = if a.is_class {
            (a.baseline_error - e) * 100.0
        } else {
            a.baseline_error.sqrt() - e.sqrt()
        };
        if diff >= 0.0 {
            format!("+{:.4}", diff)
        } else {
            format!("{:.4}", diff)
        }
    };

    eprintln!("\n=== {} (n={}, max_k={}) ===", name, a.n, a.max_k);
    eprintln!(
        "  Baseline (hard k={}, 1/d):     {}",
        a.baseline_k,
        fmt(a.baseline_error)
    );
    eprintln!(
        "  Gaussian fixed h={:.4}:        {} ({})",
        a.gaussian_fixed_h,
        fmt(a.gaussian_fixed_error),
        gain(a.gaussian_fixed_error)
    );
    eprintln!(
        "  Gaussian adaptive k_ref={}:     {} ({})",
        a.gaussian_adaptive_k_ref,
        fmt(a.gaussian_adaptive_error),
        gain(a.gaussian_adaptive_error)
    );
    eprintln!(
        "  Tricube fixed h={:.4}:         {} ({})",
        a.tricube_fixed_h,
        fmt(a.tricube_fixed_error),
        gain(a.tricube_fixed_error)
    );
    eprintln!(
        "  Tricube adaptive k_ref={}:      {} ({})",
        a.tricube_adaptive_k_ref,
        fmt(a.tricube_adaptive_error),
        gain(a.tricube_adaptive_error)
    );
    eprintln!("  Oracle (per-point best):       {}", fmt(a.oracle_error));
    eprintln!("  ---");
    eprintln!("  Headroom (baseline→oracle):    {}", gain(a.oracle_error));
}

#[test]
fn soft_weighting_iris() {
    let data = load_csv(include_str!("../testdata/iris.csv"));
    let a = analyze_soft_weighting(&data);
    print_soft_analysis("Iris", &a);
}

#[test]
fn soft_weighting_wine() {
    let data = load_csv_target(include_str!("../testdata/wine.csv"), Some(0));
    let a = analyze_soft_weighting(&data);
    print_soft_analysis("Wine", &a);
}

#[test]
fn soft_weighting_auto_mpg() {
    let data = load_csv_target(include_str!("../testdata/auto_mpg.csv"), Some(0));
    let a = analyze_soft_weighting(&data);
    print_soft_analysis("Auto MPG", &a);
}

#[test]
fn soft_weighting_ionosphere() {
    let data = load_csv(include_str!("../testdata/ionosphere.csv"));
    let a = analyze_soft_weighting(&data);
    print_soft_analysis("Ionosphere", &a);
}

#[test]
fn soft_weighting_breast_cancer() {
    let data = load_csv(include_str!("../testdata/breast_cancer.csv"));
    let a = analyze_soft_weighting(&data);
    print_soft_analysis("Breast Cancer", &a);
}
