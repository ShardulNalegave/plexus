
pub mod relu;
pub mod linear;
pub mod sigmoid;
pub mod softmax;

// ===== Imports =====
// ===================

pub trait ActivationFunction {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64>;
}