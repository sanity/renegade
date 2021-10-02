use pav_regression::pav::IsotonicRegression;

pub trait Labelled {
    fn label(&self) -> &str;
}

pub trait Metric<InputType> {
    fn distance(&self, input_a: InputType, input_b: InputType) -> f64;
}

pub trait Row<InputType, OutputType> {
    fn input(&self) -> InputType;
    fn output(&self) -> OutputType;
}

pub trait Learner {
    type InputType;
    type OutputType;
    type InstanceType: Row<Self::InputType, Self::OutputType>;
    type TrainingDataType: IntoIterator<Item = Self::InstanceType>;
    type MetricType: Metric<Self::InputType> + Labelled;

    fn learn_metrics(
        &self,
        training_data: Self::TrainingDataType,
        input_metrics: Vec<Box<Self::MetricType>>,
        output_metric: Box<dyn Metric<Self::OutputType>>,
    ) -> Self::MetricType {
        let mut metrics : Vec<LearnedMetric<Self::InputType>> = vec!();
        
        todo!();
    }
}

struct LearnedMetric<InputType> {
    regression : IsotonicRegression,
    inputMetric : Box<dyn Metric<InputType>>,
}