
// ===== Imports =====
use crate::loss::LossFunction;
// ===================

pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy {
  fn calculate(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> (na::DMatrix<f64>, f64) {
    unimplemented!()
  }

  fn derivative(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    unimplemented!()
  }
}