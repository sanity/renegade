use bit_vec::BitVec;
use rand::prelude::*;

#[derive(Clone)]
pub struct WaypointIndex<I: Clone + PartialEq<I>> {
    waypoints: Vec<(I, I)>,
    dist: Box<fn(&I, &I) -> f64>,
}

impl<I: Clone + PartialEq<I>> WaypointIndex<I> {
    pub fn new(
        data: &Vec<I>,
        dist: fn(&I, &I) -> f64,
        waypoint_count: usize,
        sample_count: usize,
        rng: &mut ThreadRng,
    ) -> WaypointIndex<I> {
        let samples = data
            .choose_multiple(rng, sample_count)
            .map(|x| x.clone())
            .collect();

        let mut waypoints: Vec<(I, I)> = vec![];
        let mut waypoint_samples: Vec<BitVec> = vec![BitVec::new(); waypoint_count];
        for _ in 0..waypoint_count {
            let mut first_best: Option<((I, I), f64, BitVec)> = Option::None;
            for _ in 0..sample_count {
                let waypoint_candidate = Self::random_distance_pair(rng, data);
                let correlations: BitVec = Self::calc_sides(dist, &samples, &waypoint_candidate);
                let score = calc_score(&waypoint_samples, &correlations);
                if first_best.is_none() || first_best.as_ref().unwrap().1 < score {
                    first_best = Option::Some((waypoint_candidate, score, correlations));
                }
            }

            match first_best {
                Some((waypoint, _, c)) => {
                    waypoints.push(waypoint);
                    waypoint_samples.push(c);
                }
                None => {
                    panic!("Unable to find waypoint")
                }
            }
        }
        WaypointIndex {
            waypoints,
            dist: Box::new(dist),
        }
    }

    pub fn calc_lsh(&self, item: &I) -> BitVec {
        let mut bv = BitVec::from_elem(self.waypoints.len(), false);
        for (ix, wp_pair) in self.waypoints.iter().enumerate() {
            bv.set(ix, Self::waypoint_side(*self.dist, wp_pair, item));
        }
        bv
    }

    fn random_distance_pair(rng: &mut ThreadRng, vec: &Vec<I>) -> (I, I) {
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

    fn select_random(rng: &mut ThreadRng, vec: &Vec<I>) -> I {
        vec[rng.gen_range(0..vec.len())].clone()
    }

}

fn calc_score(correlations_by_waypoint: &Vec<BitVec>, target_correlations: &BitVec) -> f64 {
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
    fn calc_score_test() {

    }

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
