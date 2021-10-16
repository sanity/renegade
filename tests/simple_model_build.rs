extern crate renegade;

use rand::prelude::*;
use renegade::LearnerConfig;

#[test]
fn simple_model_build() {
    let mut rng = thread_rng();
    let mut data: Vec<((f64, f64), f64)> = vec![];
    for _ in 0..1000 {
        //data.push((rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)));
    }
    
    let input_metrics : fn(&(f64, f64), &(f64, f64)) -> Vec<f64> = | a, b | 
        vec![(a.0-b.0).abs(), (a.1-b.1).abs()];
    ;

    let wi = renegade::metric::learn_metrics(data, input_metrics, |a : f64, b : f64| {(a-b).abs()}, &LearnerConfig {
        sample_count : 100,
        train_test_prop : 0.5,
        iterations : 10,
    });
}
