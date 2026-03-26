# Renegade

A nonparametric metric learning library in Rust. Given data with inputs and outputs, Renegade learns a distance function over inputs that predicts how similar their outputs are.

## How It Works

### Learned Metric (`metric.rs`)

The core of the library. Takes pairs of data points, computes multiple input distance features between them, and uses **isotonic regression** (PAV algorithm) to learn a monotonic mapping from each input distance to an estimated output distance. The final distance is the sum of all per-feature regressions.

The learning process:
1. Split data into train/test sets
2. Sample random pairs and compute input distances + output distances
3. Fit initial isotonic regressions (one per input feature)
4. Iteratively refine using residual correction with a learning rate
5. Evaluate RMSE on the test set each iteration

### Waypoint Index (`index.rs`)

A locality-sensitive hashing (LSH) scheme for approximate nearest neighbor search. Random pairs of data points serve as "waypoints" — each item is classified by which member of the pair it's closer to, producing a compact `BitVec` hash. Waypoints are greedily selected to maximize even splitting and minimize correlation with existing waypoints.

### Optimizer (`opt.rs`)

A stub hyperparameter optimizer (not yet implemented), plus a `Dist` type that does piecewise-linear interpolation sampling from a sorted set of values.

### Model (`lib.rs`)

A `Model` struct combining `LearnedMetric` + `WaypointIndex`. The struct is defined but the impl is empty — the pieces haven't been wired together yet.

## Status

Early prototype. The metric learning and waypoint index components exist independently but aren't yet connected into a unified model. The optimizer is unimplemented.

### Known Issues

- **Bug in `index.rs`**: `calc_score` creates a zero-length vector (`vec![len; 0]` instead of `vec![0; len]`), causing a panic on index access.
- **Bug in `index.rs`**: Waypoint selection uses `max_by_key` on a `BitVec` (lexicographic comparison) instead of comparing the score.
- **Unnecessary `RwLock` contention**: `calculate_point_vectors` takes a write lock inside a parallel `for_each`, likely serializing the work.
- **Stale idioms**: `extern crate` declarations (unnecessary since Rust 2018), `&Vec<T>` instead of `&[T]`, mismatched edition settings in `Cargo.toml`.

## Usage

```rust
use renegade::metric::learn_metrics;
use renegade::LearnerConfig;

let config = LearnerConfig {
    sample_count: 10000,
    train_test_prop: 0.5,
    iterations: 10,
    learning_rate: 0.01,
};

// Data: Vec<(InputType, OutputType)>
let data: Vec<(Vec<f64>, f64)> = /* your data */;

// Define how to measure distances between inputs and outputs
fn input_metrics(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).collect()
}

fn output_metric(a: &f64, b: &f64) -> f64 {
    (a - b).abs()
}

let regressions = learn_metrics(&data, input_metrics, output_metric, &config);
```

## Dependencies

- `pav_regression` — Isotonic (pool adjacent violators) regression
- `rayon` — Parallel iteration
- `bit-vec` — Compact bitwise hashing for the waypoint index
- `ordered-float` — Ordered floats for the optimizer
- `indicatif` — Progress bars
- `tracing` / `env_logger` — Logging
- `rand` — Random sampling
- `xz` — Decompression for test data
