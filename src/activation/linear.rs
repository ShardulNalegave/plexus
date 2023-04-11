
// ===== Imports =====
use crate::activation::ActivationFunction;
// ===================

pub struct Linear;

impl ActivationFunction for Linear {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    inputs.clone()
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    gradients.clone()
  }
}