
// ===== Imports =====
use crate::{
  output_layer_utils::CombinedActivationLossFunction,
  activation,
  loss,
};
use crate::activation::ActivationFunction;
use crate::loss::LossFunction;
// ===================

#[allow(non_camel_case_types)]
pub struct ActivationSoftmax_LossCategoricalCrossEntropy {
  activation: activation::Softmax,
  loss: loss::CategoricalCrossEntropy,
}

impl Default for ActivationSoftmax_LossCategoricalCrossEntropy {
  fn default() -> Self {
    Self {
      activation: activation::Softmax::default(),
      loss: loss::CategoricalCrossEntropy,
    }
  }
}

impl CombinedActivationLossFunction for ActivationSoftmax_LossCategoricalCrossEntropy {
  fn forward(&mut self, inputs: &na::DMatrix<f64>, expected_outputs: &na::DMatrix<f64>) -> (na::DVector<f64>, f64) {
    let activation_outputs = self.activation.forward(inputs);
    self.loss.calculate(&activation_outputs, expected_outputs)
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>, expected_outputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let num_samples = gradients.nrows() as f64;

    gradients.map_with_location(|i, j, x| {
      (x - expected_outputs[(i,j)]) / num_samples
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_backward_pass() {
    let softmax_outputs = na::DMatrix::from_vec(3, 3, vec![
      0.7, 0.1, 0.02,
      0.1, 0.5, 0.9,
      0.2, 0.4, 0.08,
    ]);
    let targets = na::DMatrix::from_vec(3, 3, vec![
      1.0, 0.0, 0.0,
      0.0, 1.0, 1.0,
      0.0, 0.0, 0.0,
    ]);

    let mut comb_func = ActivationSoftmax_LossCategoricalCrossEntropy::default();
    let gradients = comb_func.backward(&softmax_outputs, &targets);

    println!("{}", softmax_outputs);
    println!("{}", targets);
    println!("{}", gradients);

    assert!(true);
  }
}