
// ===== Imports =====
use crate::activation::ActivationFunction;
// ===================

pub struct Step;

impl ActivationFunction for Step {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    inputs.map(|x| {
      if x > 0.0 {
        1.0
      } else {
        0.0
      }
    })
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    na::DMatrix::zeros(gradients.nrows(), gradients.ncols())
  }
}