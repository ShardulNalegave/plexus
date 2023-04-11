
mod categorical_cross_entropy;

pub use self::{
  categorical_cross_entropy::CategoricalCrossEntropy,
};

// ===== Imports =====
use crate::descriptor::LossType;
// ===================

pub trait LossFunction {
  fn calculate(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> (na::DMatrix<f64>, f64);
  fn derivative(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}

pub fn loss_function_from_descriptor(loss_type: LossType) -> Box<dyn LossFunction> {
  match loss_type {
    LossType::CategoricalCrossEntropy => Box::new(CategoricalCrossEntropy),
  }
}