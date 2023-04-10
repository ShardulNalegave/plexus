
pub mod categorical_cross_entropy;

#[derive(Copy, Clone)]
pub enum Loss {
  CategoricalCrossEntropy,
}

impl Loss {
  pub fn calculate(&self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> (na::DMatrix<f64>, f64) {
    match *self {
      Self::CategoricalCrossEntropy
        => categorical_cross_entropy::CategoricalCrossEntropy::calculate(predicted, expected),
    }
  }

  pub fn derivative(&self, predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    match *self {
      Self::CategoricalCrossEntropy
        => categorical_cross_entropy::CategoricalCrossEntropy::derivative(predicted, expected),
    }
  }
}