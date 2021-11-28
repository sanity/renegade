use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::prelude::*;
use std::collections::*;

enum Parameter {}

struct Optimizer {
    scores: BTreeMap<OrderedFloat<f64>, Vec<HashMap<String, Parameter>>>,
    recommended: HashSet<HashMap<String, Parameter>>,
}

impl Optimizer {
    fn new() -> Optimizer {
        todo!();
    }

    fn report_score(&mut self, config: HashMap<String, Parameter>, score: f64) {
        self.scores
            .entry(OrderedFloat(score))
            .or_insert_with(Vec::new)
            .push(config);
    }
}

impl Iterator for Optimizer {
    type Item = HashMap<String, Parameter>;

    fn next(&mut self) -> Option<HashMap<String, Parameter>> {
        todo!();
    }
}

pub struct Dist<V: PartialOrd<V>> {
    values: Vec<V>,
}

impl<V: PartialOrd<V> + Clone> Dist<V> {
    pub fn new<VS: Into<Vec<V>>>(values: VS) -> Dist<V> {
        let mut values = values.into();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Dist { values }
    }

    pub fn values(&self) -> Vec<V> {
        self.values.clone()
    }
}

impl Distribution<f64> for Dist<f64> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let dist = &self.values;
        let s: f64 = rng.gen_range(0.0..(dist.len() - 1) as f64);
        let ix = s.trunc() as usize;
        let before = dist[ix];
        let after = dist[ix + 1];
        let fract = s.fract();
        (before * (1.0 - fract)) + (after * fract)
    }
}
