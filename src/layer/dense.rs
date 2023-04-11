
// ===== Imports =====
use rand::{Rng, thread_rng};
use crate::layer::Layer;
// ===================

pub struct Dense {
  weights: na::DMatrix<f64>,
  biases: na::DVector<f64>,
  inputs: Option<na::DMatrix<f64>>,
}

impl Dense {
  pub fn new(num_neurons: usize, num_weights: usize) -> Self {
    let weights = na::DMatrix::from_fn(num_neurons, num_weights, |_, _| thread_rng().gen());
    let biases = na::DVector::from_element(num_neurons, 1.0);

    Self { weights, biases, inputs: None }
  }

  pub fn get_weights(&self) -> &na::DMatrix<f64> {
    &self.weights
  }

  pub fn get_biases(&self) -> &na::DVector<f64> {
    &self.biases
  }
}

impl Layer for Dense {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inputs = Some(inputs.clone());
    (inputs * &self.weights.transpose()).map_with_location(|i, _, x| {
      x + self.biases[i]
    })
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let inputs = self.inputs.as_ref()
      .expect("Backward pass called before Forward pass");

    let _dweights = inputs.transpose() * gradients;
    let _dbiases = gradients.row_sum();
    let dinputs = gradients * &self.weights;

    dinputs
  }
}