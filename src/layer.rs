
pub mod dense;

// ===== Imports =====
use ganit::{Matrix, RVector};
// ===================

pub trait Layer<const N: usize, const I: usize> {
  fn weights(&self) -> &Matrix<N, I, f64>;
  fn bias(&self) -> &RVector<N, f64>;

  fn forward<const B: usize>(&self, inputs: Matrix<B, I, f64>) -> Matrix<B, N, f64>;
}