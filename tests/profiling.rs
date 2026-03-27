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

fn make_dataset(n: usize, d: usize, seed: u64) -> Vec<(ProfilePoint, f64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ranges = vec![(0.0, 1.0); d];
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| rng.gen()).collect();
            let output: f64 = features.iter().take(3.min(d)).sum();
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
fn profile_training_and_inference() {
    eprintln!();
    eprintln!("=== Performance Profile ===");
    eprintln!(
        "{:<8} {:<8} {:<15} {:<15} {:<15} {:<15}",
        "n", "d", "train (ms)", "query 1 (µs)", "query 100 (ms)", "predict (µs)"
    );
    eprintln!("{}", "-".repeat(80));

    for &(n, d) in &[
        (100, 5),
        (500, 5),
        (1000, 5),
        (1000, 20),
        (5000, 5),
        (5000, 20),
        (10000, 5),
    ] {
        let data = make_dataset(n, d, 42);
        let query_point = data[0].0.clone();

        let mut model = Renegade::new();
        for (p, o) in &data {
            model.add(p.clone(), *o);
        }

        let t0 = Instant::now();
        model.get_optimal_k();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Warm up
        let _ = model.query_k(&query_point, 5);

        let t0 = Instant::now();
        let _ = model.query_k(&query_point, 5);
        let query_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        let t0 = Instant::now();
        for i in 0..100 {
            let _ = model.query_k(&data[i % data.len()].0, 5);
        }
        let query_100_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let _ = model.predict_k(&query_point, 5);
        let predict_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        eprintln!(
            "{:<8} {:<8} {:<15.1} {:<15.0} {:<15.1} {:<15.0}",
            n, d, train_ms, query_us, query_100_ms, predict_us
        );
    }
    eprintln!();

    // Detailed query breakdown at n=5000, d=5
    eprintln!("=== Query Breakdown (n=5000, d=5, avg of 100 iters) ===");
    let data = make_dataset(5000, 5, 42);
    let mut model = Renegade::new();
    for (p, o) in &data {
        model.add(p.clone(), *o);
    }
    model.get_optimal_k();

    let query = &data[0].0;
    let iters = 100;

    // Warm up
    for _ in 0..10 {
        let _ = model.query_k(query, 5);
    }

    // Full query_k
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = model.query_k(query, 5);
    }
    let query_k_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Just feature_values() on query (1 alloc)
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = query.feature_values();
    }
    let fv_query_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // feature_distances for all 5000 points (5000 allocs)
    let t0 = Instant::now();
    for _ in 0..iters {
        for (p, _) in &data {
            let _ = query.feature_distances(p);
        }
    }
    let fd_all_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Raw distance computation without Vec allocation (inline arithmetic)
    let qf = &query.features;
    let ranges = &query.ranges;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut _total = 0.0f64;
        for (p, _) in &data {
            let mut sum = 0.0;
            for k in 0..5 {
                sum += (qf[k] - p.features[k]).abs() / (ranges[k].1 - ranges[k].0);
            }
            _total += sum / 5.0;
        }
    }
    let raw_dist_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Sort of 5000 (usize, f64) tuples
    let dummy: Vec<(usize, f64)> = (0..5000).map(|i| (i, (i as f64 * 17.3) % 1.0)).collect();
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut d = dummy.clone();
        d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let sort_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Vec<(usize, f64)> collect from iterator
    let t0 = Instant::now();
    for _ in 0..iters {
        let _v: Vec<(usize, f64)> = (0..5000).map(|i| (i, i as f64 * 0.001)).collect();
    }
    let collect_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!("  Full query_k:                    {:.3} ms", query_k_ms);
    eprintln!("  feature_values() (1 call):       {:.3} ms", fv_query_ms);
    eprintln!("  feature_distances() x5000:       {:.3} ms", fd_all_ms);
    eprintln!("  Raw inline distance x5000:       {:.3} ms", raw_dist_ms);
    eprintln!(
        "  Vec alloc overhead (fd - raw):    {:.3} ms",
        fd_all_ms - raw_dist_ms
    );
    eprintln!("  Sort 5000 tuples:                {:.3} ms", sort_ms);
    eprintln!("  Collect 5000 tuples:             {:.3} ms", collect_ms);
    eprintln!(
        "  Unaccounted:                     {:.3} ms",
        query_k_ms - fv_query_ms - fd_all_ms - sort_ms - collect_ms
    );
    eprintln!();

    // Training breakdown
    eprintln!("=== Training Breakdown (n=5000, d=5) ===");
    let data = make_dataset(5000, 5, 42);
    let mut model = Renegade::new();
    for (p, o) in &data {
        model.add(p.clone(), *o);
    }
    let t0 = Instant::now();
    model.get_optimal_k();
    let full_train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    model.force_retrain();
    let t0 = Instant::now();
    model.get_optimal_k();
    let retrain_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  First train:  {:.1} ms", full_train_ms);
    eprintln!("  Retrain:      {:.1} ms", retrain_ms);
    eprintln!();

    // VP-tree profiling
    eprintln!("=== VP-Tree vs Brute Force ===");
    eprintln!(
        "{:<8} {:<8} {:<15} {:<15} {:<15}",
        "n", "d", "build (ms)", "vp query (µs)", "bf query (µs)"
    );
    eprintln!("{}", "-".repeat(65));

    for &(n, d) in &[(1000, 5), (5000, 5), (10000, 5), (50000, 5), (100000, 5)] {
        let data = make_dataset(n, d, 42);
        let ranges = vec![(0.0, 1.0); d];

        // Build VP-tree
        let dist_fn = |a: usize, b: usize| -> f64 {
            data[a]
                .0
                .features
                .iter()
                .zip(data[b].0.features.iter())
                .zip(ranges.iter())
                .map(|((x, y), (lo, hi))| (x - y).abs() / (hi - lo))
                .sum::<f64>()
                / d as f64
        };

        let t0 = Instant::now();
        let tree = renegade_ml::vptree::VpTree::build(n, &dist_fn);
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // VP-tree query (average of 100)
        let iters = 100;
        let t0 = Instant::now();
        for qi in 0..iters {
            let query_dist = |i: usize| -> f64 { dist_fn(qi % n, i) };
            let _ = tree.query_nearest(5, &query_dist);
        }
        let vp_query_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        // Brute-force query (average of 100)
        let t0 = Instant::now();
        for qi in 0..iters {
            let mut dists: Vec<(usize, f64)> = (0..n).map(|i| (i, dist_fn(qi % n, i))).collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(5);
        }
        let bf_query_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

        eprintln!(
            "{:<8} {:<8} {:<15.1} {:<15.0} {:<15.0}",
            n, d, build_ms, vp_query_us, bf_query_us
        );
    }
    eprintln!();
}
