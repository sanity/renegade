# Renegade

A nonparametric supervised learning library for Rust. Zero configuration, competitive with scikit-learn KNN out of the box.

Renegade is a KNN-based learner that **just works** — no hyperparameters to tune, no preprocessing pipeline to configure. It handles mixed numeric and categorical features, automatically selects K via leave-one-out cross-validation, and optionally learns a distance metric that downweights irrelevant features.

## Features

- **Zero parameters** — K is auto-selected, metric learning is automatic, no tuning required
- **Mixed feature types** — numeric and categorical features handled natively via Gower distance
- **Learned metric** — per-feature isotonic regressions learn which features matter, with automatic fallback when the metric doesn't help
- **Multiple output modes** — nearest neighbors, weighted mean, class probabilities, distance-trend extrapolation
- **Incremental insertion** — add data points without rebuilding; expensive retraining is amortized (triggers only when data doubles)
- **Designed for small to medium datasets** — up to ~100k points with brute-force search

## Quick Start

```rust
use renegade::{DataPoint, Renegade};

// Define your data type
#[derive(Clone)]
struct MyPoint {
    temperature: f64,
    pressure: f64,
    category: u8,
}

impl DataPoint for MyPoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        vec![
            (self.temperature - other.temperature).abs() / 100.0,  // normalized to [0,1]
            (self.pressure - other.pressure).abs() / 50.0,
            if self.category == other.category { 0.0 } else { 1.0 },
        ]
    }

    fn feature_values(&self) -> Vec<f64> {
        vec![self.temperature, self.pressure, self.category as f64]
    }

    fn num_features(&self) -> usize { 3 }
}

// Build and query
let mut model = Renegade::new();
model.add(MyPoint { temperature: 20.0, pressure: 1013.0, category: 1 }, 0.85);
model.add(MyPoint { temperature: 35.0, pressure: 1005.0, category: 2 }, 0.42);
// ... add more data ...

// Auto-selects K, learns metric if beneficial
let prediction = model.predict(&query_point);
println!("Predicted: {:.2} (R²: {:.2})", prediction.value, prediction.r_squared);

// Or get raw neighbors for custom aggregation
let neighbors = model.query(&query_point);
let class_probs = neighbors.class_votes();       // classification
let mean = neighbors.weighted_mean();             // regression
let sample = neighbors.sample(rand_value);        // random draw
```

## How It Works

### Layer 1: Gower Distance + Auto K

Each feature contributes a distance in [0, 1]:
- **Numeric**: `|a - b| / range`
- **Categorical**: `0` if same, `1` if different

The composite distance is the mean of per-feature distances. K is selected automatically via leave-one-out cross-validation — no default K that might be wrong for your data.

### Layer 2: Effect-Space Metric Learning

For each feature, an isotonic regression maps feature values to their marginal effect on the output. Features that predict the output get high weight; noise features get zero weight. Distances are then computed in this "effect space."

The metric is only used when it demonstrably improves LOO prediction error. If it doesn't help (e.g., all features are equally informative), the system automatically falls back to the simple Gower distance. This means **the metric never hurts**.

### Distance-Trend Extrapolation

Instead of just averaging neighbor outputs, Renegade can fit a linear trend of output vs distance and extrapolate to distance=0. This handles cases where the query point is outside the training data distribution, providing a prediction with an R² confidence score.

## Benchmark Results

Leave-one-out cross-validation against scikit-learn's KNN (best configuration with StandardScaler and tuned K):

| Dataset | n | Features | Renegade | sklearn best |
|---------|---|----------|----------|-------------|
| Iris | 150 | 4 | 94.0% | 96.7% |
| Wine | 178 | 13 | **98.9%** | 96.6% |
| Auto MPG | 393 | 7 (mixed) | **RMSE 2.78** | RMSE 2.88 |
| Ionosphere | 351 | 34 | **98.6%** | 91.2% |
| Breast Cancer | 569 | 30 | **97.4%** | 96.7% |
| Wine Quality | 4898 | 11 | **RMSE 0.73** | RMSE 0.74 |

Renegade wins 5 of 6 datasets with zero configuration vs sklearn requiring choice of scaler, distance metric, and K.

## Design Constraints

- **No hyperparameters** — every configuration option is an opportunity for misconfiguration
- **No multivariate optimization** — no gradient descent, no learning rates, no convergence concerns
- **Brute-force search** — designed for up to ~100k data points. For larger datasets, consider a data retention strategy (sliding window, exponential decay)
- **Training cost is amortized** — metric learning and K selection only recompute when the dataset has doubled in size

## Intended Use Cases

- **Online learning** with moderate data volumes (hundreds to low thousands of points)
- **Mixed-type data** where features are numeric, categorical, or custom (edit distance, Jaccard, etc.)
- **Low-data regimes** where parametric models overfit
- **Routing and scheduling** decisions based on historical performance (e.g., peer selection in distributed systems)

## License

AGPL-3.0-or-later
