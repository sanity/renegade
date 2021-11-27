extern crate bit_vec;
extern crate env_logger;
extern crate indicatif;
extern crate log;
extern crate pav_regression;
extern crate tracing;
extern crate rand;
extern crate ordered_float;
extern crate chrono;

pub mod index;
pub mod metric;
pub mod opt;

use index::*;
use metric::*;

pub struct Model<InputType, OutputType>
where
    InputType: Clone + PartialEq<InputType> + ?Sized + Send + Sync,
    OutputType: Clone,
{
    metric: LearnedMetric<InputType, OutputType>,
    index: WaypointIndex<InputType>,
}

impl<InputType, OutputType> Model<InputType, OutputType>
where
    InputType: Clone + PartialEq<InputType> + ?Sized + Send + Sync,
    OutputType: Clone,
{

}

pub trait Labelled {
    fn label(&self) -> &str;
}

pub struct LearnerConfig {
    pub sample_count: usize,
    pub train_test_prop: f64,
    pub iterations : usize,
    pub learning_rate: f64,
}

#[cfg(test)]
mod tests {

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn it_works() {
        init();
        assert_eq!(2 + 2, 4);
    }
}
