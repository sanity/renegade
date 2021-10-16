use std::sync::RwLock;

use pav_regression::pav::{IsotonicRegression, Point};
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::*;

pub fn learn_metrics<InputType, OutputType>(
    data: &Vec<(InputType, OutputType)>,
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    config: &LearnerConfig,
) -> RwLock<Vec<IsotonicRegression>>
where
    InputType: Copy,
    OutputType: Copy,
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

fn sample_distances<InputType, OutputType>(
    rng: &mut ThreadRng,
    training_data: &Vec<(InputType, OutputType)>,
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    sample_count: usize,
) -> Vec<(Vec<f64>, f64)>
where
    InputType: Copy,
    OutputType: Copy,
{
    assert!(sample_count < training_data.len() * (training_data.len() - 1));

    let mut samples: Vec<(Vec<f64>, f64)> = vec![];
    while samples.len() < sample_count {
        let a_ix = rng.gen_range(0..training_data.len());
        let b_ix = rng.gen_range(0..training_data.len());
        if a_ix != b_ix {
            let a = &training_data[a_ix];
            let b = &training_data[b_ix];
            let output_distance = output_metric(&a.1, &b.1);
            let mut input_distances: Vec<f64> = input_metrics(&a.0, &b.0);
            samples.push((input_distances, output_distance));
        }
    }
    samples
}

fn create_initial_regressions(distance_samples: &Vec<(Vec<f64>, f64)>) -> Vec<IsotonicRegression> {
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

    use crate::metric::*;

    #[test]
    fn split_train_test_test() {
        let mut rng = thread_rng();
        let data = gen_simple_data(&mut rng);
        let (train, test): (Vec<(f64, f64)>, Vec<(f64, f64)>) =
            split_train_test(&mut rng, 0.5, &data);
        assert!((20..70).contains(&train.len()));
        assert!((20..70).contains(&test.len()));
        assert!(train.len() + test.len() == 100);
    }

    #[test]
    fn sample_distances_test() {
        let mut rng = thread_rng();
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
        let mut rng = thread_rng();
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

    fn gen_simple_data(rng: &mut ThreadRng) -> Vec<(f64, f64)> {
        let mut data: Vec<(f64, f64)> = vec![];
        for _ in 0..100 {
            let v = rng.gen_range(0.0..1.0);
            data.push((v, v));
        }
        data
    }
}
