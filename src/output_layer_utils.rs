
mod softmax_categorical_cross_entropy;

pub use self::{
  softmax_categorical_cross_entropy::ActivationSoftmax_LossCategoricalCrossEntropy,
};

pub fn combined_activation_loss_function_from_descriptor() {
  //
}

pub trait CombinedActivationLossFunction {
  fn forward(&mut self, inputs: &na::DMatrix<f64>, expected_outputs: &na::DMatrix<f64>) -> (na::DVector<f64>, f64);
  fn backward(&mut self, gradients: &na::DMatrix<f64>, expected_outputs: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}