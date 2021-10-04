use std::sync::RwLock;

use pav_regression::pav::{IsotonicRegression, Point};
use rand::{prelude::ThreadRng, thread_rng, Rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::*;

pub(crate) fn learn_metrics<InputType, OutputType, MetricType>(
    data: &Vec<(InputType, OutputType)>,
    input_metrics: &Vec<Box<MetricType>>,
    output_metric: &Box<dyn Metric<OutputType>>,
    config: &LearnerConfig,
) -> RwLock<Vec<IsotonicRegression>>
where
    InputType: Copy,
    OutputType: Copy,
    MetricType: Metric<InputType> + Sync,
{
    let mut rng = thread_rng();

    let (training_data, testing_data) = split_train_test(&mut rng, config.train_test_prop, data);

    let training_samples = sample_distances(
        &mut rng,
        &training_data,
        input_metrics,
        output_metric,
        config.sample_count,
    );
    let testing_samples = sample_distances(
        &mut rng,
        &testing_data,
        input_metrics,
        output_metric,
        config.sample_count,
    );

    let mut regressions = RwLock::new(create_initial_regressions(&training_samples));

    for iteration in 0..config.iterations {
        refine_regressions(&training_samples, &mut regressions);
        let rmse = test_regressions(&testing_samples, &regressions);
        println!("#{}\r{}", iteration, rmse);
    }

    regressions
}

fn sample_distances<InputType, OutputType, MetricType>(
    rng: &mut ThreadRng,
    training_data: &Vec<(InputType, OutputType)>,
    input_metrics: &Vec<Box<MetricType>>,
    output_metric: &Box<dyn Metric<OutputType>>,
    sample_count: usize,
) -> Vec<(Vec<f64>, f64)>
where
    InputType: Copy,
    OutputType: Copy,
    MetricType: Metric<InputType> + Sync,
{
    assert!(sample_count < training_data.len() * (training_data.len() - 1));

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

fn calculate_point_vectors(
    distance_samples: &Vec<(Vec<f64>, f64)>,
    regressions: &RwLock<Vec<IsotonicRegression>>,
) -> RwLock<Vec<Vec<Point>>> {
    let num_input_metrics = distance_samples.first().unwrap().0.len();
    let point_vectors: RwLock<Vec<Vec<Point>>> = RwLock::new(vec![vec![]; num_input_metrics]);
    distance_samples
        .par_iter()
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
                point_vectors.write().unwrap()[ix]
                    .push(Point::new(input_distances[ix], correction));
            }
        });
    point_vectors
}

fn create_initial_regressions(distance_samples: &Vec<(Vec<f64>, f64)>) -> Vec<IsotonicRegression> {
    let num_input_metrics = distance_samples.first().unwrap().0.len();
    distance_samples
        .par_iter()
        .map(|(input_distances, output_distance)| {
            let mut points: Vec<Point> = vec![];
            for input_dist in input_distances {
                points.push(Point::new(
                    *input_dist,
                    output_distance / num_input_metrics as f64,
                ));
            }
            IsotonicRegression::new_ascending(&points)
        })
        .collect()
}

fn refine_regressions(
    distance_samples: &Vec<(Vec<f64>, f64)>,
    regressions: &mut RwLock<Vec<IsotonicRegression>>,
) {
    let point_vectors = calculate_point_vectors(&distance_samples, &regressions);
    let refined_regressions: Vec<IsotonicRegression> = point_vectors
        .read()
        .unwrap()
        .par_iter()
        .map(|points| -> IsotonicRegression { IsotonicRegression::new_ascending(&points) })
        .collect();

    let mut writable_regressions = regressions.write().unwrap();
    *writable_regressions = refined_regressions;
}

fn split_train_test<InputType, OutputType>(
    rng: &mut ThreadRng,
    train_test_prop: f64,
    data: &Vec<(InputType, OutputType)>,
) -> (Vec<(InputType, OutputType)>, Vec<(InputType, OutputType)>)
where
    InputType: Copy,
    OutputType: Copy,
{
    let mut training_data: Vec<(InputType, OutputType)> = vec![];
    let mut testing_data: Vec<(InputType, OutputType)> = vec![];

    for &d in data {
        if rng.gen_bool(train_test_prop) {
            training_data.push(d);
        } else {
            testing_data.push(d);
        }
    }

    (training_data, testing_data)
}

fn test_regressions(
    distance_samples: &Vec<(Vec<f64>, f64)>,
    regressions: &RwLock<Vec<IsotonicRegression>>,
) -> f64 {
    let reg = regressions.read().unwrap();
    let sum_error_squared: f64 = distance_samples
        .par_iter()
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
    let rmse = (sum_error_squared / distance_samples.len() as f64).sqrt();
    rmse
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::renegade::learn_metrics::*;

    #[test]
    fn split_train_test_test() {
        let mut rng = thread_rng();
        let data = {
            let rng: &mut ThreadRng = &mut rng;
            let mut data: Vec<(f64, f64)> = vec![];
            for _ in 0..100 {
                data.push((rng.gen(), rng.gen()));
            }
            data
        };
        let (train, test): (Vec<(f64, f64)>, Vec<(f64, f64)>) =
            split_train_test(&mut rng, 0.5, &data);
        assert!((35..55).contains(&train.len()));
        assert!((35..55).contains(&test.len()));
        assert!(train.len() + test.len() == 100);
    }

    #[test]
    fn sample_distances_test() {
        let mut rng = thread_rng();
        let data = {
            let rng: &mut ThreadRng = &mut rng;
            let mut data: Vec<((f64, f64), f64)> = vec![];
            for _ in 0..100 {
                let a: f64 = rng.gen_range(0.0..1.0);
                let b: f64 = rng.gen_range(0.0..1.0);

                data.push(((a, b), a * b));
            }
            data
        };
    }

    impl Metric<(f64, f64)> for (f64, f64) {
        fn distance(&self, input_a: &(f64, f64), input_b: &(f64, f64)) -> f64 {
            todo!()
        }
    }
}
