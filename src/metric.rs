use std::sync::RwLock;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use log::info;
use pav_regression::pav::{IsotonicRegression, Point};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::*;

use super::*;

pub struct LearnedMetric<InputType, OutputType>
where
    InputType: Clone,
    OutputType: Clone,
{
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    regressions: RwLock<Vec<IsotonicRegression>>,
}

impl<InputType, OutputType> LearnedMetric<InputType, OutputType>
where
    InputType: Clone,
    OutputType: Clone,
{
    fn new(
        data: &[(InputType, OutputType)],
        input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
        output_metric: fn(&OutputType, &OutputType) -> f64,
        config: &LearnerConfig,
    ) -> LearnedMetric<InputType, OutputType> {
        let regressions = learn_metrics(data, input_metrics, output_metric, config);
        LearnedMetric {
            input_metrics,
            output_metric,
            regressions,
        }
    }

    fn distance(&self, a: &InputType, b: &InputType) -> f64 {
        let input_distances = (self.input_metrics)(a, b);
        let mut output = 0.0;
        let regressions = self.regressions.read().unwrap();
        for (ix, distance) in input_distances.iter().enumerate() {
            output += regressions[ix].interpolate(distance);
        }
        output
    }
}

pub fn learn_metrics<InputType, OutputType>(
    data: &[(InputType, OutputType)],
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    config: &LearnerConfig,
) -> RwLock<Vec<IsotonicRegression>>
where
    InputType: Clone,
    OutputType: Clone,
{
    let _span = span!(Level::INFO, "learn_metrics", data_len = data.len()).entered();

    let mut rng = SmallRng::from_entropy();

    let (training_data, testing_data) = split_train_test(&mut rng, config.train_test_prop, data);

    let training_samples = {
        let _span = span!(
            Level::INFO,
            "sample training distances",
            training_data.len = training_data.len()
        )
        .entered();
        sample_distances(
            &mut rng,
            &training_data,
            input_metrics,
            output_metric,
            config.sample_count,
        )
    };

    let testing_samples = {
        let _span = span!(
            Level::INFO,
            "sample testing distances",
            testing_data.len = testing_data.len()
        )
        .entered();
        sample_distances(
            &mut rng,
            &testing_data,
            input_metrics,
            output_metric,
            config.sample_count,
        )
    };

    let mut regressions = RwLock::new(create_initial_regressions(&training_samples));
    {
        for iteration in 0..config.iterations {
            let _span = span!(
                Level::INFO,
                "refining regressions",
                iteration = iteration,
                iterations = config.iterations
            )
            .entered();
            refine_regressions(&training_samples, &mut regressions, config.learning_rate);
            let rmse = test_regressions(&testing_samples, &regressions);
            event!(Level::INFO, iteration = iteration, rmse = rmse);
        }
    }
    regressions
}

type TrainTestData<InputType, OutputType> =
    (Vec<(InputType, OutputType)>, Vec<(InputType, OutputType)>);

fn split_train_test<InputType, OutputType>(
    rng: &mut SmallRng,
    train_test_prop: f64,
    data: &[(InputType, OutputType)],
) -> TrainTestData<InputType, OutputType>
where
    InputType: Clone,
    OutputType: Clone,
{
    let mut training_data: Vec<(InputType, OutputType)> = vec![];
    let mut testing_data: Vec<(InputType, OutputType)> = vec![];

    for d in data {
        if rng.gen_bool(train_test_prop) {
            training_data.push(d.clone());
        } else {
            testing_data.push(d.clone());
        }
    }

    (training_data, testing_data)
}

fn sample_distances<InputType, OutputType>(
    rng: &mut SmallRng,
    training_data: &[(InputType, OutputType)],
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    sample_count: usize,
) -> Vec<(Vec<f64>, f64)>
where
    InputType: Clone,
    OutputType: Clone,
{
    let _span = span!(Level::INFO, "sample_distances", sample_count = sample_count).entered();
    assert!(sample_count < training_data.len() * (training_data.len() - 1));
    let mut samples: Vec<(Vec<f64>, f64)> = vec![];
    for _ in (0..sample_count).progress() {
        let sample_pair: Vec<&(InputType, OutputType)> =
            training_data.choose_multiple(rng, 2).collect();
        let output_distance = output_metric(&sample_pair[0].1, &sample_pair[1].1);
        let input_distances: Vec<f64> = input_metrics(&sample_pair[0].0, &sample_pair[1].0);
        samples.push((input_distances, output_distance));
    }
    samples
}

fn create_initial_regressions(distance_samples: &[(Vec<f64>, f64)]) -> Vec<IsotonicRegression> {
    let num_inputs = distance_samples.first().unwrap().0.len();
    let mut points: Vec<Vec<Point>> = vec![vec![]; num_inputs];
    for (input_distances, output_distance) in distance_samples {
        for (ix, input_dist) in input_distances.iter().enumerate() {
            let point = Point::new(*input_dist, output_distance / num_inputs as f64);
            points[ix].push(point);
        }
    }
    points
        .par_iter()
        .map(|points| IsotonicRegression::new_ascending(points))
        .collect()
}

fn calculate_point_vectors(
    distance_samples: &[(Vec<f64>, f64)],
    regressions: &RwLock<Vec<IsotonicRegression>>,
    learning_rate: f64,
) -> RwLock<Vec<Vec<Point>>> {
    let _span = span!(
        Level::INFO,
        "calculate_point_vectors",
        regressions.len = regressions.read().unwrap().len(),
    )
    .entered();
    let num_input_metrics = distance_samples.first().unwrap().0.len();
    let point_vectors: RwLock<Vec<Vec<Point>>> = RwLock::new(vec![vec![]; num_input_metrics]);
    distance_samples
        .par_iter()
        .progress_with(
            ProgressBar::new(distance_samples.len() as u64)
                .with_message("calculating point vectors across samples"),
        )
        .for_each(|(input_distances, actual_output)| {
            let estimated_output_parts: Vec<f64> = input_distances
                .iter()
                .enumerate()
                .map(|(ix, input_dist)| {
                    let regression: &IsotonicRegression = &regressions.read().unwrap()[ix];
                    regression.interpolate(input_dist)
                })
                .collect();

            let estimated_output: f64 = estimated_output_parts.iter().sum();

            for (ix, part) in estimated_output_parts.iter().enumerate() {
                let estimated_output_without_this = estimated_output - part;
                let correction = actual_output - estimated_output_without_this;
                let lr_correction = (correction * learning_rate) + (part * (1.0 - learning_rate));
                point_vectors.write().unwrap()[ix]
                    .push(Point::new(input_distances[ix], lr_correction));
            }
        });
    point_vectors
}

fn refine_regressions(
    distance_samples: &[(Vec<f64>, f64)],
    regressions: &mut RwLock<Vec<IsotonicRegression>>,
    learning_rate: f64,
) {
    let _span = span!(
        Level::INFO,
        "refine_regressions",
        distance_samples.len = distance_samples.len(),
        regressions.len = regressions.read().unwrap().len(),
    )
    .entered();
    let lock = calculate_point_vectors(distance_samples, regressions, learning_rate);
    let point_vectors = lock.read().unwrap();
    let refined_regressions: Vec<IsotonicRegression> = point_vectors
        .par_iter()
        .progress()
        .map(|points| -> IsotonicRegression { IsotonicRegression::new_ascending(points) })
        .collect();

    let mut writable_regressions = regressions.write().unwrap();
    *writable_regressions = refined_regressions;
}

fn test_regressions(
    distance_samples: &[(Vec<f64>, f64)],
    regressions: &RwLock<Vec<IsotonicRegression>>,
) -> f64 {
    info!("test_regressions");
    let reg = regressions.read().unwrap();
    let sum_error_squared: f64 = distance_samples
        .par_iter()
        .progress()
        .map(|(input_distances, output_distance)| {
            let prediction = input_distances
                .iter()
                .enumerate()
                .map(|(ix, input_dist)| reg[ix].interpolate(input_dist))
                .sum::<f64>();
            let error = prediction - output_distance;
            error * error
        })
        .sum();
    (sum_error_squared / distance_samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::metric::*;

    #[test]
    fn split_train_test_test() {
        let mut rng = SmallRng::from_entropy();
        let data = gen_simple_data(&mut rng);
        let (train, test): (Vec<(f64, f64)>, Vec<(f64, f64)>) =
            split_train_test(&mut rng, 0.5, &data);
        assert!((20..70).contains(&train.len()));
        assert!((20..70).contains(&test.len()));
        assert!(train.len() + test.len() == 100);
    }

    #[test]
    fn sample_distances_test() {
        let mut rng = SmallRng::from_entropy();
        let data = gen_simple_data(&mut rng);

        let samples = sample_distances(
            &mut rng,
            &data,
            float_input_metric,
            float_output_metric,
            100,
        );

        assert_eq!(samples.len(), 100);
        for sample in samples {
            assert_eq!(sample.0[0], sample.1);
        }
    }

    #[test]
    fn create_initial_regressions_test() {
        let mut rng = SmallRng::from_entropy();
        let data = gen_simple_data(&mut rng);

        let samples = sample_distances(
            &mut rng,
            &data,
            float_input_metric,
            float_output_metric,
            100,
        );
        let initial_regressions = create_initial_regressions(&samples);
        assert_eq!(initial_regressions.len(), 1);

        for point in initial_regressions[0].get_points() {
            // Verify that input and output distances are the same
            assert_eq!(point.x(), point.y());
        }
    }

    fn float_input_metric(a: &f64, b: &f64) -> Vec<f64> {
        vec![(a - b).abs()]
    }

    fn float_output_metric(a: &f64, b: &f64) -> f64 {
        (a - b).abs()
    }

    fn gen_simple_data(rng: &mut SmallRng) -> Vec<(f64, f64)> {
        let mut data: Vec<(f64, f64)> = vec![];
        for _ in 0..100 {
            let v = rng.gen_range(0.0..1.0);
            data.push((v, v));
        }
        data
    }
}
