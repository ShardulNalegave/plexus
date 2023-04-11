
// ===== Imports =====
use crate::activation::ActivationFunction;
// ===================

pub struct Softmax {
  outputs: Option<na::DMatrix<f64>>,
}

impl Default for Softmax {
  fn default() -> Self {
    Self {
      outputs: None,
    }
  }
}

impl ActivationFunction for Softmax {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let max_values = na::DVector::from_fn(inputs.nrows(), |i, _| inputs.row(i).max());
    let adjusted_inputs = inputs - max_values;

    let expo_inputs = adjusted_inputs.map(|x| f64::exp(x));
    let sum_vector = expo_inputs.column_sum();

    let outputs = na::DMatrix::from_fn(expo_inputs.nrows(), expo_inputs.ncols(), |i, j| {
      expo_inputs[(i,j)] / sum_vector[i]
    });

    self.outputs = Some(outputs.clone());
    outputs
  }

  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let outputs = self.outputs.as_ref()
      .expect("Backward pass called before Forward pass");

    let mut derivative = na::DMatrix::zeros(gradients.nrows(), gradients.ncols());

    for sample_num in 0..gradients.nrows() {
      let sample_gradients = gradients.row(sample_num).clone_owned();
      let sample_outputs = outputs.row(sample_num).clone_owned().transpose();

      let jacobian_matrix = (na::DMatrix::from_diagonal(&sample_outputs) - (&sample_outputs * sample_outputs.transpose())).clone_owned();
      let dvalues = sample_gradients * jacobian_matrix;

      derivative.set_row(sample_num, &dvalues);
    }

    derivative
  }
}