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

    fn num_features(&self) -> usize {
        2
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

    fn num_features(&self) -> usize {
        2
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
    let pred = model.predict_k(&Point2D::new(0.0, 0.0, range, range), 20);
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
    assert_eq!(class_0_prob, 0.75);
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

    let pred = model.predict_k(&Point2D::new(0.0, 0.0, range, range), 10);
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
    let pred = model.predict_k(&Point2D::new(0.0, 0.0, range, range), 2);
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
