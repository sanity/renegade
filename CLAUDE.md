# Renegade - Claude Code Configuration

## Project

Renegade is a zero-configuration KNN library for Rust. Key source files:
- `src/lib.rs` — core algorithm, K selection, training orchestration
- `src/metric.rs` — isotonic regression metric learning
- `src/vptree.rs` — VP-tree nearest neighbor index
- `src/neighbor.rs` — neighbor aggregation (weighted mean, class votes)
- `src/predict.rs` — extrapolated prediction
- `tests/benchmarks.rs` — standard ML dataset benchmarks
