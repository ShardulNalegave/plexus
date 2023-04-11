
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
    gradients.map_with_location(|i, j, x| {
      x - expected_outputs[(i,j)]
    })
  }
}