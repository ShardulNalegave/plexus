
// ===== Imports =====
use ganit::{CVector, Matrix};
use rand::Rng;
use crate::layer::Layer;
// ===================

#[derive(Clone, Copy, Debug)]
pub struct DenseLayer<const N: usize, const I: usize> {
  weights: Matrix<N, I, f64>,
  bias: CVector<N, f64>,
}

impl<const N: usize, const I: usize> DenseLayer<N, I> {
  pub fn new() -> Self {
    let mut rng = rand::thread_rng();
    let bias: CVector<N, f64> = CVector::new([[0.0; 1]; N]);
    let weights: Matrix<N, I, f64> = Matrix::new_with_gen(|(_, _)| rng.gen());

    Self { weights, bias }
  }
}

impl<const N: usize, const I: usize> Layer<N, I> for DenseLayer<N, I> {
  fn weights(&self) -> &Matrix<N, I, f64> { &self.weights }
  fn bias(&self) -> &CVector<N, f64> { &self.bias }
  
  fn forward(&self, inputs: CVector<I, f64>) -> CVector<N, f64> {
    let mut outputs = [[0_f64; 1]; N];

    for i in 0..N {
      let mut output = 0.0;
      let weights = self.weights.row(i);
      let bias = self.bias[(i, 0)];

      for j in 0..I {
        output += weights[j] * inputs[(j, 0)];
      }

      output += bias;
      outputs[i] = [output];
    }

    CVector::new(outputs)
  }
}