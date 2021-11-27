use std::sync::RwLock;

use bit_vec::BitVec;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::*;

#[derive(Clone)]
pub struct WaypointIndex<I: Clone + PartialEq<I> + Send + Sync> {
    waypoints: Vec<(I, I)>,
    dist: Box<fn(&I, &I) -> f64>,
}

impl<I: Clone + PartialEq<I> + ?Sized + Send + Sync> WaypointIndex<I> {
    pub fn new(
        data: &[I],
        dist: fn(&I, &I) -> f64,
        num_waypoints: usize,
        num_item_samples: usize,
        num_waypoint_samples: usize,
    ) -> WaypointIndex<I> {
        let mut rng = SmallRng::from_entropy();
        let samples = data
            .choose_multiple(&mut rng, num_item_samples)
            .cloned()
            .collect();

        let waypoints: RwLock<Vec<(I, I)>> = RwLock::new(vec![]);
        let waypoint_samples: RwLock<Vec<BitVec>> = RwLock::new(vec![BitVec::new(); num_waypoints]);
        for waypoint_no in 0..num_waypoints {
            let best = repeatn((), num_waypoint_samples)
                .map(|_| {
                    let waypoint_candidate = Self::random_distance_pair(data);
                    let correlations: BitVec =
                        Self::calc_sides(dist, &samples, &waypoint_candidate);
                    let score = calc_score(&waypoint_samples.read().unwrap(), &correlations);
                    (waypoint_candidate, score, correlations)
                })
                .max_by_key(|f| f.2.clone());

            match best {
                Some((waypoint, _, c)) => {
                    waypoints.write().unwrap().push(waypoint);
                    waypoint_samples.write().unwrap().push(c);
                }
                None => {
                    panic!("Unable to find {}th waypoint", waypoint_no);
                }
            }
        }
        // let is needed due to borrower issue I don't understand
        let waypoint_index = WaypointIndex {
            waypoints: waypoints.read().unwrap().clone(),
            dist: Box::new(dist),
        };
        waypoint_index
    }

    pub fn calc_lsh(&self, item: &I) -> BitVec {
        let mut bv = BitVec::from_elem(self.waypoints.len(), false);
        for (ix, wp_pair) in self.waypoints.iter().enumerate() {
            bv.set(ix, Self::waypoint_side(*self.dist, wp_pair, item));
        }
        bv
    }

    fn random_distance_pair(vec: &[I]) -> (I, I) {
        let mut rng = SmallRng::from_entropy();
        loop {
            let a = vec[rng.gen_range(0..vec.len())].clone();
            let b = vec[rng.gen_range(0..vec.len())].clone();
            if a != b {
                return (a, b);
            }
        }
    }

    /// For a given waypoint determine which side of each sample it's on, results returned as BitVec
    /// same size as samples.len()
    fn calc_sides(dist: fn(&I, &I) -> f64, samples: &Vec<I>, waypoint: &(I, I)) -> BitVec {
        let mut bv = BitVec::new();
        for sample in samples {
            bv.push(Self::waypoint_side(dist, waypoint, sample));
        }
        bv
    }

    /// Determine which side of a waypoint an item is on
    fn waypoint_side(dist: fn(&I, &I) -> f64, waypoint: &(I, I), item: &I) -> bool {
        let d1 = (dist)(&waypoint.0, item);
        let d2 = (dist)(&waypoint.1, item);
        d1 < d2
    }
}

fn calc_score(correlations_by_waypoint: &[BitVec], target_correlations: &BitVec) -> f64 {
    let mut same_counts_by_waypoint: Vec<usize> = vec![correlations_by_waypoint.len(); 0];
    for (waypoint_ix, waypoint_correlations) in correlations_by_waypoint.iter().enumerate() {
        for (sample_ix, target_cor) in target_correlations.iter().enumerate() {
            let sample_side_same_as_waypoint = !(waypoint_correlations[sample_ix] ^ target_cor);
            if sample_side_same_as_waypoint {
                same_counts_by_waypoint[waypoint_ix] += 1;
            }
        }
    }

    let true_count: usize = target_correlations.iter().filter(|c| *c).count();
    let separation_score = count_to_score(true_count, target_correlations.len());

    let correlation_score = same_counts_by_waypoint
        .iter()
        .map(|s| count_to_score(*s, target_correlations.len()))
        .min_by(|a, b| a.partial_cmp(b).unwrap());

    match correlation_score {
        None => separation_score,
        Some(cs) => cs.min(separation_score),
    }
}

fn count_to_score(true_count: usize, total: usize) -> f64 {
    let s = (true_count as f64) / (total as f64);
    if s > 0.5 {
        1.0 - s
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::WaypointIndex;

    #[test]
    fn calc_score_test() {}

    #[test]
    fn calc_correlations_test() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let waypoint = (2.4, 2.6);
        let correlations =
            WaypointIndex::calc_sides(|a, b| ((a - b) as f64).abs(), &samples, &waypoint);
        assert_eq!(correlations.len(), 4);
        assert_eq!(correlations[0], true);
        assert_eq!(correlations[1], true);
        assert_eq!(correlations[2], false);
        assert_eq!(correlations[3], false);
    }

    #[test]
    fn count_to_score_test() {
        assert_eq!(count_to_score(4, 10), 0.4);
        assert_eq!(count_to_score(6, 10), 0.4);
    }
}

