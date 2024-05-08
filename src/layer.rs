
pub mod dense;

// ===== Imports =====
use ganit::{CVector, Matrix};
// ===================

pub trait Layer<const N: usize, const I: usize> {
  fn weights(&self) -> &Matrix<N, I, f64>;
  fn bias(&self) -> &CVector<N, f64>;

  fn forward(&self, inputs: CVector<I, f64>) -> CVector<N, f64>;
}