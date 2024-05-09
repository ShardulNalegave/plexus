
// ===== Imports =====
use ganit::Matrix;
use super::ActivationFunction;
// ===================

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64> {
    Matrix::new_with_gen(|(i, j)| 1.0 / (1.0 + (-inputs[(i, j)]).exp()))
  }
}