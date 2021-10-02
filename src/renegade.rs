use pav_regression::pav::{IsotonicRegression, Point};
use rand::Rng;
use rayon::prelude::*;
use std::sync::RwLock;

pub trait Labelled {
    fn label(&self) -> &str;
}

pub trait Metric<InputType> {
    fn distance(&self, input_a: &InputType, input_b: &InputType) -> f64;
}

pub trait Learner {
    type InputType;
    type OutputType;
    type MetricType: Metric<Self::InputType> + Labelled + Sync;

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
        let mut initial_regressions = RwLock::new(Self::create_initial_regressions(distance_samples));

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

    fn create_initial_regressions(
        distance_samples : Vec<(Vec<f64>, f64)>,
    ) -> Vec<IsotonicRegression> {
        let num_input_metrics = distance_samples.first().unwrap().0.len();
        distance_samples.par_iter().map(|(input_distances, output_distance)| {
            let mut points : Vec<Point> = vec!();
            for input_dist in input_distances {
                points.push(Point::new(*input_dist, output_distance / num_input_metrics as f64));
            }
            IsotonicRegression::new_ascending(&points)
        }).collect()
    }

    fn refine_regressions(
        distance_samples : Vec<(Vec<f64>, f64)>,
        regressions : RwLock<&mut Vec<IsotonicRegression>>,
    ) -> f64 {
        let num_input_metrics = distance_samples.first().unwrap().0.len();
        let points_array : RwLock<Vec<Vec<Point>>> = RwLock::new(vec![vec![]; num_input_metrics]);
        distance_samples.par_iter().for_each(|(input_distances, actual_output)| {
            let estimated_output_parts : Vec<f64> = input_distances.iter().enumerate().map(|(ix, input_dist)| {
                let regression : &IsotonicRegression = &regressions.read().unwrap()[ix];
                regression.interpolate(input_dist)
            }).collect();

            let estimated_output : f64 = estimated_output_parts.iter().sum();

            for (ix, part) in estimated_output_parts.iter().enumerate() {
                let estimated_output_without_this = estimated_output-part;
                let correction = actual_output - estimated_output_without_this;
                points_array.write().unwrap()[ix].push(Point::new(input_distances[ix], correction));
            }
            

        });

        let newRegressions : Vec<IsotonicRegression> = points_array.read().unwrap().par_iter().map(|points| -> IsotonicRegression {
            IsotonicRegression::new_ascending(&points)
        }
        ).collect();

        let mut r = regressions.write().unwrap();
        r.clear();
        for (ix, new_regression) in newRegressions.iter().enumerate() {
            r[ix] = new_regression.clone();
        }

        todo!();
    }

}

pub struct LearnerConfig {
    sample_count: usize,
}
