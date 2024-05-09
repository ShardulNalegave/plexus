
// ===== Imports =====
use ganit::Matrix;
use super::ActivationFunction;
// ===================

pub struct ReLU;
impl ActivationFunction for ReLU {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64> {
    Matrix::new_with_gen(|(i, j)| inputs[(i, j)].max(0.0))
  }
}