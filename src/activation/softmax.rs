
// ===== Imports =====
use ganit::{CVector, Matrix};
use super::ActivationFunction;
// ===================

pub struct Softmax;
impl ActivationFunction for Softmax {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64> {
    let exponentiated_inputs: Matrix<B, N, f64> = Matrix::new_with_gen(|(i, j)| inputs[(i, j)].exp());
    let input_wise_summation: Matrix<B, 1, f64> = CVector::new_with_gen(|(i, _)| {
      let mut sum = 0.0;
      for j in 0..N {
        sum += exponentiated_inputs[(i, j)];
      }
      sum
    });

    Matrix::new_with_gen(|(i, j)| {
      exponentiated_inputs[(i, j)] / input_wise_summation[(i, 1)]
    })
  }
}