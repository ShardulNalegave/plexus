
// ===== Imports =====
use crate::activation::ActivationFunction;
// ===================

pub struct Sigmoid {
  inputs: Option<na::DMatrix<f64>>,
}

impl Default for Sigmoid {
  fn default() -> Self {
    Self {
      inputs: None,
    }
  }
}

impl ActivationFunction for Sigmoid {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inputs = Some(inputs.clone());
    inputs.map(|x| {
      let expo = f64::exp(-x);
      1.0 / (1.0 + expo)
    })
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let inputs = self.inputs.as_ref()
      .expect("Backward pass called before Forward pass");

    gradients.map_with_location(|i, j, gradient| {
      let expo = f64::exp(-inputs[(i,j)]);
      let dsigmoid = (-expo) / (1.0 + expo).powi(2);

      gradient * dsigmoid
    })
  }
}
