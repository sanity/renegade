use pav_regression::pav::{IsotonicRegression, Point};
use rand::Rng;
use rayon::prelude::*;

pub trait Labelled {
    fn label(&self) -> &str;
}

pub trait Metric<InputType> {
    fn distance(&self, input_a: &InputType, input_b: &InputType) -> f64;
}

pub trait Learner {
    type InputType;
    type OutputType;
    type MetricType: Metric<Self::InputType> + Labelled;

    fn learn_metrics(
        training_data: &Vec<(Self::InputType, Self::OutputType)>,
        input_metrics: &Vec<Box<Self::MetricType>>,
        output_metric: &Box<dyn Metric<Self::OutputType>>,
        config: &LearnerConfig,
    ) -> Self::MetricType {
        let distance_samples = Self::sample_distances(
            training_data,
            input_metrics,
            output_metric,
            config.sample_count,
        );
        let mut initial_points: Vec<Vec<Point>> = vec![vec!(); input_metrics.len()];
        for (inputs, output) in distance_samples {
            for input_ix in 0..input_metrics.len() {
                initial_points[input_ix].push(Point::new(inputs[input_ix], output));
            }
        }
        todo!();
    }

    fn sample_distances(
        training_data: &Vec<(Self::InputType, Self::OutputType)>,
        input_metrics: &Vec<Box<Self::MetricType>>,
        output_metric: &Box<dyn Metric<Self::OutputType>>,
        sample_count: usize,
    ) -> Vec<(Vec<f64>, f64)> {
        assert!(sample_count < training_data.len() * (training_data.len() - 1));

        let mut rng = rand::thread_rng();
        let mut samples: Vec<(Vec<f64>, f64)> = vec![];
        while samples.len() < sample_count {
            let a_ix = rng.gen_range(0..training_data.len());
            let b_ix = rng.gen_range(0..training_data.len());
            if a_ix != b_ix {
                let a = &training_data[a_ix];
                let b = &training_data[b_ix];
                let output_distance = output_metric.distance(&a.1, &b.1);
                let mut input_distances: Vec<f64> = vec![];
                for metric in input_metrics {
                    input_distances.push(metric.distance(&a.0, &b.0));
                }
                samples.push((input_distances, output_distance));
            }
        }
        samples
    }
}

pub struct LearnerConfig {
    sample_count: usize,
}
