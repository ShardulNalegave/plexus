
pub mod linear;
pub mod step;
pub mod relu;
pub mod softmax;
pub mod sigmoid;

#[derive(Copy, Clone)]
pub enum Activation {
  Linear,
  Step,
  ReLU,
  Softmax,
  Sigmoid,
}

impl Activation {
  pub fn forward(&self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    match *self {
      Self::Step => step::Step::forward(inputs),
      Self::Linear => linear::Linear::forward(inputs),
      Self::ReLU => relu::ReLU::forward(inputs),
      Self::Softmax => softmax::Softmax::forward(inputs),
      Self::Sigmoid => sigmoid::Sigmoid::forward(inputs),
    }
  }

  pub fn backward(&self, gradients: &na::DMatrix<f64>, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    match *self {
      Self::Step => step::Step::backward(gradients, inputs),
      Self::Linear => linear::Linear::backward(gradients, inputs),
      Self::ReLU => relu::ReLU::backward(gradients, inputs),
      Self::Softmax => softmax::Softmax::backward(gradients, inputs),
      Self::Sigmoid => sigmoid::Sigmoid::backward(gradients, inputs),
    }
  }
}