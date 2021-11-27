extern crate renegade;

use log::LevelFilter;
use rand::prelude::*;
use renegade::metric::learn_metrics;
use renegade::*;
use tracing::*;

fn init() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(LevelFilter::Info)
        .try_init();
}

#[test]
fn simple_model_build() {
    init();

    let mut data = vec![];

    let mut rng = thread_rng();
    for _ in 0..1000 {
        let inputs = vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];
        data.push((inputs.clone(), true_function(&inputs)));
    }

    let input_metrics = |a: &Vec<f64>, b: &Vec<f64>| vec![(a[0] - b[0]).abs(), (a[1] - b[1]).abs()];
    let output_metric = |a: &f64, b: &f64| (a - b).abs();
    let config = LearnerConfig {
        sample_count: 10000,
        train_test_prop: 0.5,
        iterations: 10,
        learning_rate: 0.01,
    };
    learn_metrics(&data, input_metrics, output_metric, &config);
}

fn true_function(inputs: &Vec<f64>) -> f64 {
    ((inputs[0] - 0.5) * (inputs[0] - 0.5)) + ((inputs[1] - 0.5) * (inputs[1] - 0.5))
}
