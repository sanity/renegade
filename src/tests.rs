use crate::neighbor::{Neighbor, Neighbors};
use crate::{DataPoint, Renegade};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Simple 2D numeric point for testing.
#[derive(Clone, Debug)]
struct Point2D {
    x: f64,
    y: f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
}

impl Point2D {
    fn new(x: f64, y: f64, x_range: (f64, f64), y_range: (f64, f64)) -> Self {
        Point2D {
            x,
            y,
            x_range,
            y_range,
        }
    }
}

impl DataPoint for Point2D {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        let dx = (self.x - other.x).abs() / (self.x_range.1 - self.x_range.0);
        let dy = (self.y - other.y).abs() / (self.y_range.1 - self.y_range.0);
        vec![dx, dy]
    }

    fn feature_values(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }
}

/// Mixed numeric + categorical point for testing.
#[derive(Clone, Debug)]
struct MixedPoint {
    value: f64,
    value_range: (f64, f64),
    category: String,
}

impl DataPoint for MixedPoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        let numeric_dist =
            (self.value - other.value).abs() / (self.value_range.1 - self.value_range.0);
        let cat_dist = if self.category == other.category {
            0.0
        } else {
            1.0
        };
        vec![numeric_dist, cat_dist]
    }

    fn feature_values(&self) -> Vec<f64> {
        // Categorical encoded as a numeric value for metric learning
        let cat_val = match self.category.as_str() {
            "A" => 0.0,
            "B" => 1.0,
            _ => 0.5,
        };
        vec![self.value, cat_val]
    }
}

#[test]
fn exact_match_returns_correct_output() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);
    model.add(Point2D::new(5.0, 5.0, range, range), 42.0);
    model.add(Point2D::new(0.0, 0.0, range, range), 10.0);
    model.add(Point2D::new(10.0, 10.0, range, range), 100.0);

    let neighbors = model.query_k(&Point2D::new(5.0, 5.0, range, range), 1);
    assert_eq!(neighbors.neighbors.len(), 1);
    assert_eq!(neighbors.neighbors[0].output, 42.0);
    assert_eq!(neighbors.neighbors[0].distance, 0.0);
}

#[test]
fn weighted_mean_with_exact_match() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);
    model.add(Point2D::new(5.0, 5.0, range, range), 42.0);
    model.add(Point2D::new(0.0, 0.0, range, range), 10.0);

    let neighbors = model.query_k(&Point2D::new(5.0, 5.0, range, range), 2);
    // Exact match should dominate weighted mean.
    assert_eq!(neighbors.weighted_mean(), 42.0);
}

#[test]
fn linear_function_extrapolation() {
    // output = 2*x + 3*y, query at origin should predict ~0
    let mut model = Renegade::new();
    let range = (0.0, 10.0);
    let mut rng = SmallRng::seed_from_u64(42);

    for _ in 0..200 {
        let x: f64 = rng.gen_range(0.5..10.0);
        let y: f64 = rng.gen_range(0.5..10.0);
        let output = 2.0 * x + 3.0 * y;
        model.add(Point2D::new(x, y, range, range), output);
    }

    // Query near origin — extrapolation should predict close to 0.
    let pred = model.predict_k_extrapolated(&Point2D::new(0.0, 0.0, range, range), 20);
    assert!(
        pred.value.abs() < 5.0,
        "Expected prediction near 0, got {}",
        pred.value
    );
}

#[test]
fn categorical_feature_separates_classes() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);

    // Category A -> output ~10, Category B -> output ~90
    let mut rng = SmallRng::seed_from_u64(123);
    for _ in 0..50 {
        let v: f64 = rng.gen_range(4.0..6.0);
        model.add(
            MixedPoint {
                value: v,
                value_range: range,
                category: "A".into(),
            },
            10.0 + rng.gen_range(-1.0..1.0),
        );
        model.add(
            MixedPoint {
                value: v,
                value_range: range,
                category: "B".into(),
            },
            90.0 + rng.gen_range(-1.0..1.0),
        );
    }

    // Query category A — should predict near 10.
    let neighbors = model.query_k(
        &MixedPoint {
            value: 5.0,
            value_range: range,
            category: "A".into(),
        },
        10,
    );
    let mean = neighbors.weighted_mean();
    assert!(
        (mean - 10.0).abs() < 5.0,
        "Expected ~10 for category A, got {}",
        mean
    );

    // Query category B — should predict near 90.
    let neighbors = model.query_k(
        &MixedPoint {
            value: 5.0,
            value_range: range,
            category: "B".into(),
        },
        10,
    );
    let mean = neighbors.weighted_mean();
    assert!(
        (mean - 90.0).abs() < 5.0,
        "Expected ~90 for category B, got {}",
        mean
    );
}

#[test]
fn class_votes_returns_correct_probabilities() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);

    // 3 class-0 points near origin, 1 class-1 point nearby.
    model.add(Point2D::new(0.0, 0.0, range, range), 0.0);
    model.add(Point2D::new(0.1, 0.1, range, range), 0.0);
    model.add(Point2D::new(0.2, 0.2, range, range), 0.0);
    model.add(Point2D::new(0.3, 0.3, range, range), 1.0);

    let neighbors = model.query_k(&Point2D::new(0.0, 0.0, range, range), 4);
    let votes = neighbors.class_votes();

    let class_0_prob = votes.iter().find(|(c, _)| *c == 0.0).unwrap().1;
    // With distance-weighted voting, class 0 should dominate
    // (one point at distance 0, two more nearby, vs one class-1 farther away)
    assert!(
        class_0_prob > 0.75,
        "Class 0 should have >75% weighted vote, got {:.3}",
        class_0_prob
    );
}

#[test]
fn r_squared_indicates_fit_quality() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);

    // Perfect linear relationship with distance.
    for i in 1..=10 {
        let v = i as f64;
        model.add(Point2D::new(v, 0.0, range, range), v * 2.0);
    }

    let pred = model.predict_k_extrapolated(&Point2D::new(0.0, 0.0, range, range), 10);
    assert!(
        pred.r_squared > 0.9,
        "Expected high R² for linear data, got {}",
        pred.r_squared
    );
}

#[test]
fn small_dataset_still_works() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);

    model.add(Point2D::new(1.0, 1.0, range, range), 10.0);
    model.add(Point2D::new(9.0, 9.0, range, range), 90.0);

    // With only 2 points, should still give a prediction.
    let pred = model.predict_k_extrapolated(&Point2D::new(0.0, 0.0, range, range), 2);
    assert!(!pred.value.is_nan());
    assert_eq!(pred.k, 2);
}

#[test]
fn auto_k_selection_works() {
    let mut model = Renegade::new();
    let range = (0.0, 10.0);
    let mut rng = SmallRng::seed_from_u64(42);

    // Two clusters with different outputs
    for _ in 0..50 {
        let x: f64 = rng.gen_range(0.0..2.0);
        let y: f64 = rng.gen_range(0.0..2.0);
        model.add(Point2D::new(x, y, range, range), 0.0);
    }
    for _ in 0..50 {
        let x: f64 = rng.gen_range(8.0..10.0);
        let y: f64 = rng.gen_range(8.0..10.0);
        model.add(Point2D::new(x, y, range, range), 1.0);
    }

    let k = model.get_optimal_k();
    eprintln!("Auto-selected K: {}", k);
    assert!(
        k >= 1 && k <= 10,
        "K={} seems unreasonable for 100 points in 2 clusters",
        k
    );

    // Should classify correctly with auto K
    let neighbors = model.query(&Point2D::new(1.0, 1.0, range, range));
    let votes = neighbors.class_votes();
    let predicted = votes
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(predicted, 0.0);
}

#[test]
fn gaussian_weighted_mean_correctness() {
    // Three neighbors at known distances with known outputs.
    // h=1.0, so w(d) = exp(-d²/2).
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.1,
                output: 10.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.5,
                output: 20.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 2.0,
                output: 30.0,
                weight: 1.0,
            },
        ],
    };
    let h = 1.0;
    let result = neighbors.gaussian_weighted_mean(h);

    // Hand-compute: w0 = exp(-0.01/2) ≈ 0.99501, w1 = exp(-0.25/2) ≈ 0.88250, w2 = exp(-4/2) ≈ 0.13534
    let w0 = (-0.01_f64 / 2.0).exp();
    let w1 = (-0.25_f64 / 2.0).exp();
    let w2 = (-4.0_f64 / 2.0).exp();
    let expected = (w0 * 10.0 + w1 * 20.0 + w2 * 30.0) / (w0 + w1 + w2);

    assert!(
        (result - expected).abs() < 1e-10,
        "Gaussian weighted mean: got {}, expected {}",
        result,
        expected
    );

    // Single neighbor: returns that neighbor's output
    let single = Neighbors {
        neighbors: vec![Neighbor {
            distance: 0.5,
            output: 42.0,
            weight: 1.0,
        }],
    };
    assert!((single.gaussian_weighted_mean(1.0) - 42.0).abs() < 1e-10);
}

#[test]
fn gaussian_weighted_mean_tiny_bandwidth_falls_back() {
    // Very small bandwidth: all weights underflow to 0, should fall back to nearest neighbor.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.1,
                output: 99.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.5,
                output: 50.0,
                weight: 1.0,
            },
        ],
    };
    let result = neighbors.gaussian_weighted_mean(1e-100);
    assert!(
        (result - 99.0).abs() < 1e-10,
        "Tiny bandwidth should fall back to nearest neighbor, got {}",
        result
    );
}

#[test]
fn gaussian_weighted_mean_exact_match() {
    // Distance 0 should be handled like weighted_mean: return exact match output.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.0,
                output: 7.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.1,
                output: 100.0,
                weight: 1.0,
            },
        ],
    };
    assert!((neighbors.gaussian_weighted_mean(1.0) - 7.0).abs() < 1e-10);
}

#[test]
fn classification_does_not_get_bandwidth() {
    let range = (0.0, 10.0);
    let mut model = Renegade::new();

    // Two clusters with integer outputs = classification
    for i in 0..50 {
        model.add(Point2D::new(i as f64 * 0.1, 0.0, range, range), 0.0);
    }
    for i in 0..50 {
        model.add(Point2D::new(5.0 + i as f64 * 0.1, 0.0, range, range), 1.0);
    }

    let _ = model.predict(&Point2D::new(0.5, 0.0, range, range));
    let diag = model.diagnostics();
    assert!(
        diag.kernel_bandwidth.is_none(),
        "Classification should not get Gaussian bandwidth, got {:?}",
        diag.kernel_bandwidth
    );
    assert!(diag.is_classification);
}

#[test]
fn integer_regression_not_misdetected_as_classification() {
    // Many distinct integer values relative to dataset size = regression, not classification.
    // e.g., ratings 1-20 with 50 data points.
    let range = (0.0, 50.0);
    let mut model = Renegade::new();
    for i in 0..50 {
        // Output is i % 20 — 20 distinct integer values out of 50 points
        // sqrt(50) ≈ 7, so 20 > 7 → should be detected as regression
        model.add(Point2D::new(i as f64, 0.0, range, range), (i % 20) as f64);
    }

    let _ = model.predict(&Point2D::new(25.0, 0.0, range, range));
    let diag = model.diagnostics();
    assert!(
        !diag.is_classification,
        "20 distinct integer values out of 50 points should be regression, not classification"
    );
}

// --- Dispersion ---

#[test]
fn dispersion_empty_returns_none() {
    let neighbors = Neighbors { neighbors: vec![] };
    assert!(neighbors.dispersion().is_none());
    assert!(neighbors.gaussian_dispersion(1.0).is_none());
}

#[test]
fn dispersion_single_neighbor_has_zero_variance() {
    let neighbors = Neighbors {
        neighbors: vec![Neighbor {
            distance: 0.7,
            output: 42.0,
            weight: 1.0,
        }],
    };
    let d = neighbors.dispersion().unwrap();
    // A single weight*output/weight round-trip can be off by an ULP or two
    // from the input, so compare with tolerance rather than bit-exactly.
    assert!((d.mean - 42.0).abs() < 1e-9, "got mean {}", d.mean);
    assert!(d.variance.abs() < 1e-15, "got variance {}", d.variance);
    assert!(d.std_dev.abs() < 1e-9, "got std_dev {}", d.std_dev);
    // A single point is its own whole (weighted) population, regardless of
    // its weight or distance.
    assert!(
        (d.effective_n - 1.0).abs() < 1e-10,
        "single neighbor should have effective_n ≈ 1, got {}",
        d.effective_n
    );

    let g = neighbors.gaussian_dispersion(1.0).unwrap();
    assert!((g.mean - 42.0).abs() < 1e-9, "got mean {}", g.mean);
    assert!(g.variance.abs() < 1e-15, "got variance {}", g.variance);
}

#[test]
fn dispersion_identical_outputs_has_zero_variance() {
    // Different distances/weights, but every output agrees exactly.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.1,
                output: 5.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.9,
                output: 5.0,
                weight: 2.0,
            },
            Neighbor {
                distance: 3.0,
                output: 5.0,
                weight: 0.3,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    assert!((d.mean - 5.0).abs() < 1e-9, "got mean {}", d.mean);
    assert!(
        d.variance.abs() < 1e-15,
        "identical outputs must have ~zero variance, got {}",
        d.variance
    );
    assert!(d.std_dev.abs() < 1e-9, "got std_dev {}", d.std_dev);

    let g = neighbors.gaussian_dispersion(1.0).unwrap();
    assert!(
        g.variance.abs() < 1e-15,
        "identical outputs must have ~zero variance, got {}",
        g.variance
    );
}

#[test]
fn dispersion_widely_disagreeing_outputs_has_large_variance() {
    // Two equally-weighted, equidistant neighbors that disagree completely.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 1.0,
                output: 0.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 1.0,
                output: 100.0,
                weight: 1.0,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    assert!((d.mean - 50.0).abs() < 1e-10);
    // Weighted population variance of {0, 100} about 50, equal weights: 2500.
    assert!(
        (d.variance - 2500.0).abs() < 1e-6,
        "expected variance 2500, got {}",
        d.variance
    );
    assert!((d.std_dev - 50.0).abs() < 1e-6);
}

#[test]
fn dispersion_effective_n_uses_kish_formula_not_raw_count() {
    // Unequal weights (same distance, so instance_weight ratios pass
    // through unscaled) make Kish's ESS diverge from a naive neighbor
    // count, which is the whole point of using it.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 1.0,
                output: 1.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 1.0,
                output: 2.0,
                weight: 3.0,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    // Kish ESS = (Σw)² / Σw² = (1+3)² / (1²+3²) = 16/10 = 1.6, NOT 2.
    assert!(
        (d.effective_n - 1.6).abs() < 1e-9,
        "expected Kish effective_n 1.6 (not raw count 2), got {}",
        d.effective_n
    );
}

#[test]
fn dispersion_variance_is_weighted_not_naive() {
    // Same distance for both (so instance_weight ratios pass through
    // unscaled), but unequal instance weights and outputs — this
    // distinguishes a correctly weighted variance from a naive
    // (unweighted) average of squared deviations, which would give a
    // different number here.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 1.0,
                output: 0.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 1.0,
                output: 100.0,
                weight: 3.0,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    // mean = (1*0 + 3*100) / 4 = 75
    assert!((d.mean - 75.0).abs() < 1e-9, "got mean {}", d.mean);
    // weighted variance = (1*(0-75)^2 + 3*(100-75)^2) / 4 = 1875
    // (a naive unweighted average of squared deviations would give 1562.5)
    assert!(
        (d.variance - 1875.0).abs() < 1e-6,
        "expected weighted variance 1875, got {}",
        d.variance
    );
}

#[test]
fn dispersion_exact_match_ignores_distant_neighbors() {
    // weighted_mean() short-circuits to exact matches only; dispersion()
    // must use the exact same subset, not the full neighbor list.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.0,
                output: 10.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.0,
                output: 10.0,
                weight: 1.0,
            },
            // Wildly different output, but at nonzero distance — must be
            // excluded from both the mean and its dispersion.
            Neighbor {
                distance: 5.0,
                output: 9999.0,
                weight: 1.0,
            },
        ],
    };
    assert_eq!(neighbors.weighted_mean(), 10.0);
    let d = neighbors.dispersion().unwrap();
    assert_eq!(d.mean, 10.0);
    assert_eq!(
        d.variance, 0.0,
        "distant neighbor must not leak into dispersion of an exact match"
    );

    let g = neighbors.gaussian_dispersion(1.0).unwrap();
    assert_eq!(g.mean, 10.0);
    assert_eq!(g.variance, 0.0);
}

#[test]
fn dispersion_mean_matches_weighted_mean_randomized() {
    // dispersion() and weighted_mean() are computed by independent code
    // paths; guard against them silently drifting apart.
    let mut rng = SmallRng::seed_from_u64(7);
    for trial in 0..200 {
        let n = 1 + (trial % 8);
        let mut neighbors = Vec::new();
        for i in 0..n {
            neighbors.push(Neighbor {
                distance: if trial % 17 == 0 && i == 0 {
                    0.0 // occasionally exercise the exact-match branch
                } else {
                    rng.gen_range(0.01..10.0)
                },
                output: rng.gen_range(-100.0..100.0),
                weight: rng.gen_range(0.1..5.0),
            });
        }
        let set = Neighbors { neighbors };
        let expected = set.weighted_mean();
        let d = set.dispersion().unwrap();
        assert!(
            (d.mean - expected).abs() < 1e-9,
            "trial {trial}: dispersion mean {} != weighted_mean {}",
            d.mean,
            expected
        );
    }
}

#[test]
fn gaussian_dispersion_mean_matches_gaussian_weighted_mean_randomized() {
    let mut rng = SmallRng::seed_from_u64(11);
    for trial in 0..200 {
        let n = 1 + (trial % 8);
        let mut neighbors: Vec<Neighbor> = (0..n)
            .map(|_| Neighbor {
                distance: rng.gen_range(0.01..3.0),
                output: rng.gen_range(-100.0..100.0),
                weight: rng.gen_range(0.1..5.0),
            })
            .collect();
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let set = Neighbors { neighbors };
        let h = rng.gen_range(0.1..2.0);
        let expected = set.gaussian_weighted_mean(h);
        match set.gaussian_dispersion(h) {
            Some(g) => assert!(
                (g.mean - expected).abs() < 1e-9,
                "trial {trial}: gaussian_dispersion mean {} != gaussian_weighted_mean {}",
                g.mean,
                expected
            ),
            None => {
                // Only valid when gaussian_weighted_mean also degenerated
                // to the nearest-neighbor fallback.
                assert_eq!(expected, set.neighbors[0].output);
            }
        }
    }
}

#[test]
fn gaussian_dispersion_tiny_bandwidth_returns_none() {
    // Mirrors gaussian_weighted_mean_tiny_bandwidth_falls_back: when no
    // neighbor contributes non-negligible weight, there is no population
    // to report dispersion over.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.1,
                output: 99.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.5,
                output: 50.0,
                weight: 1.0,
            },
        ],
    };
    assert!(neighbors.gaussian_dispersion(1e-100).is_none());
}

#[test]
fn dispersion_non_finite_output_does_not_panic() {
    // NaN/Inf outputs must propagate through the float arithmetic (as they
    // do for weighted_mean()), not panic.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 1.0,
                output: f64::NAN,
                weight: 1.0,
            },
            Neighbor {
                distance: 1.0,
                output: 5.0,
                weight: 1.0,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    assert!(d.mean.is_nan());
    assert!(d.variance.is_nan());

    // An infinite distance drives that neighbor's inverse-distance weight
    // to 0 without producing NaN, as long as its output is finite.
    let neighbors = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 1.0,
                output: 10.0,
                weight: 1.0,
            },
            Neighbor {
                distance: f64::INFINITY,
                output: 20.0,
                weight: 1.0,
            },
        ],
    };
    let d = neighbors.dispersion().unwrap();
    assert!(d.mean.is_finite());
    assert!(d.variance.is_finite());
    // The infinite-distance neighbor should carry ~0 weight, so the mean
    // should be dominated by the finite-distance neighbor.
    assert!((d.mean - 10.0).abs() < 1e-9, "got mean {}", d.mean);
}

#[test]
fn dispersion_effective_n_is_scale_invariant_to_distance() {
    // Documented caveat: Kish's effective_n does NOT decay with distance —
    // k neighbors of equal weight report effective_n ≈ k regardless of how
    // far away they are.
    let near = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 0.1,
                output: 1.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.1,
                output: 2.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 0.1,
                output: 3.0,
                weight: 1.0,
            },
        ],
    };
    let far = Neighbors {
        neighbors: vec![
            Neighbor {
                distance: 50.0,
                output: 1.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 50.0,
                output: 2.0,
                weight: 1.0,
            },
            Neighbor {
                distance: 50.0,
                output: 3.0,
                weight: 1.0,
            },
        ],
    };
    let d_near = near.dispersion().unwrap();
    let d_far = far.dispersion().unwrap();
    assert!((d_near.effective_n - 3.0).abs() < 1e-9);
    assert!(
        (d_far.effective_n - 3.0).abs() < 1e-9,
        "effective_n should stay ≈3 even for distant neighbors, got {}",
        d_far.effective_n
    );
}

#[test]
fn gaussian_dispersion_weight_sum_decays_with_distance() {
    // Unlike effective_n, the Gaussian kernel's weight_sum ("kernel mass")
    // DOES decay as the query moves away from the training data — this is
    // what makes it useful as an "is there evidence near this query" signal.
    let bandwidth = 1.0;
    let near = Neighbors {
        neighbors: vec![Neighbor {
            distance: 0.1,
            output: 1.0,
            weight: 1.0,
        }],
    };
    let far = Neighbors {
        neighbors: vec![Neighbor {
            distance: 4.0,
            output: 1.0,
            weight: 1.0,
        }],
    };
    let near_mass = near.gaussian_dispersion(bandwidth).unwrap().weight_sum;
    let far_mass = far.gaussian_dispersion(bandwidth).unwrap().weight_sum;
    assert!(
        far_mass < near_mass,
        "kernel mass should shrink with distance: near={near_mass}, far={far_mass}"
    );
    assert!(
        far_mass < 1e-3,
        "far kernel mass should be tiny, got {far_mass}"
    );
}
