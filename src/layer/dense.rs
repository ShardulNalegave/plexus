
// ===== Imports =====
use ganit::{Matrix, RVector};
use rand::Rng;
use crate::layer::Layer;
// ===================

#[derive(Clone, Copy, Debug)]
pub struct DenseLayer<const N: usize, const I: usize> {
  weights: Matrix<N, I, f64>,
  bias: RVector<N, f64>,
}

impl<const N: usize, const I: usize> DenseLayer<N, I> {
  pub fn new() -> Self {
    let mut rng = rand::thread_rng();
    let bias: RVector<N, f64> = RVector::new([[0.0; N]; 1]);
    let weights: Matrix<N, I, f64> = Matrix::new_with_gen(|(_, _)| rng.gen_range(-1.0..1.0) * 0.01);

    Self { weights, bias }
  }
}

impl<const N: usize, const I: usize> Layer<N, I> for DenseLayer<N, I> {
  fn weights(&self) -> &Matrix<N, I, f64> { &self.weights }
  fn bias(&self) -> &RVector<N, f64> { &self.bias }
  
  fn forward<const B: usize>(&self, inputs: Matrix<B, I, f64>) -> Matrix<B, N, f64> {
    (inputs * self.weights.transpose()).add_row_vector(self.bias)
  }
}