
// ===== Imports =====
use crate::loss::LossFunction;
// ===================

pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy {
  fn calculate(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> (na::DVector<f64>, f64) {
    let relevant_probabilities = predicted * expected.transpose();

    let losses = relevant_probabilities.column_sum()
      .map(|x| -clip(x, 1e-7, 1.0 - 1e-7).ln());
    let mean_loss = losses.mean();

    (losses, mean_loss)
  }

  fn derivative(&mut self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let samples = predicted.nrows();
    let labels = predicted.ncols();

    na::DMatrix::from_fn(samples, labels, |i, j| {
      let predicted_output = predicted[(i,j)];
      let expected_output = expected[(i,j)];

      let d_input = - expected_output / predicted_output;
      d_input / (samples as f64) // Normalise
    })
  }
}

fn clip(val: f64, min: f64, max: f64) -> f64 {
  val.max(min).min(max)
}