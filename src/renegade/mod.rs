mod learn_metrics;

use pav_regression::pav::IsotonicRegression;

pub fn build_model<InputType, OutputType>(
    data: &Vec<(InputType, OutputType)>,    
    input_metrics: fn(&InputType, &InputType) -> Vec<f64>,
    output_metric: fn(&OutputType, &OutputType) -> f64,
    config: &LearnerConfig,
) -> Vec<IsotonicRegression>
where
    InputType: Copy,
    OutputType: Copy,
{
    learn_metrics::learn_metrics(data, input_metrics, output_metric, config);
    todo!();
}

pub trait Labelled {
    fn label(&self) -> &str;
}

pub trait Metric<InputType> {
    fn distance(&self, input_a: &InputType, input_b: &InputType) -> f64;
}

pub struct LearnerConfig {
    sample_count: usize,
    train_test_prop: f64,
    iterations: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
}
