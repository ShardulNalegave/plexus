
// ===== Imports =====
use crate::activation::ActivationFunction;
// ===================

pub struct ReLU {
  inputs: Option<na::DMatrix<f64>>,
}

impl Default for ReLU {
  fn default() -> Self {
    Self {
      inputs: None,
    }
  }
}

impl ActivationFunction for ReLU {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inputs = Some(inputs.clone());
    inputs.map(|x| {
      f64::max(x, 0.0)
    })
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let inputs = self.inputs.as_ref()
      .expect("Backward pass called before Forward pass");

    gradients.map_with_location(|i, j, gradient| {
      if inputs[(i,j)] <= 0.0 {
        0.0
      } else {
        gradient
      }
    })
  }
}