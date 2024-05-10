
// ===== Imports =====
use ganit::{CVector, Matrix};
use super::ActivationFunction;
// ===================

pub struct Softmax;
impl ActivationFunction for Softmax {
  fn forward<const B: usize, const N: usize>(inputs: ganit::Matrix<B, N, f64>) -> ganit::Matrix<B, N, f64> {
    let exponentiated_inputs: Matrix<B, N, f64> = inputs
      .into_iter()
      .map(|val| val.exp())
      .collect();
    
    let max_values: CVector<B, f64> = CVector::new_with_gen(|(i, _)| {
      let mut max = exponentiated_inputs[(i, 0)];
      for val in exponentiated_inputs.row(i) {
        if *val > max {
          max = *val;
        }
      }
      max
    });

    let expo_normalised_inps: Matrix<B, N, f64> = exponentiated_inputs
      .into_enumerated_iter()
      .map(|(index, val)| (index, val - max_values[(index.0, 0)]))
      .collect();

    let input_wise_summation: CVector<B, f64> = CVector::new_with_gen(|(i, _)| {
      let mut sum = 0.0;
      for j in 0..N {
        sum += expo_normalised_inps[(i, j)];
      }
      sum
    });

    expo_normalised_inps.div_col_vector(input_wise_summation)
  }
}