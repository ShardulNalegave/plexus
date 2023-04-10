
pub struct CategoricalCrossEntropy;
impl CategoricalCrossEntropy {
  pub fn calculate(predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> (na::DMatrix<f64>, f64) {
    unimplemented!()
  }

  pub fn derivative(predicted: &na::DMatrix<f64>, expected: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    unimplemented!()
  }
}