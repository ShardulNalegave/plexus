
// ===== Imports =====
use super::ActivationFunction;
// ===================

pub struct Linear;
impl ActivationFunction for Linear {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64> {
    inputs
  }
}