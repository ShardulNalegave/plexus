
mod linear;
mod step;
mod relu;
mod sigmoid;
mod softmax;

pub use self::{
  linear::Linear,
  step::Step,
  relu::ReLU,
  sigmoid::Sigmoid,
  softmax::Softmax,
};

// ===== Imports =====
use crate::descriptor::ActivationType;
// ===================

pub trait ActivationFunction {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64>;
  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}

pub fn activation_function_from_descriptor(activ_type: ActivationType) -> Box<dyn ActivationFunction> {
  match activ_type {
    ActivationType::Step => Box::new(Step),
    ActivationType::Linear => Box::new(Linear),
    ActivationType::ReLU => Box::new(ReLU::default()),
    ActivationType::Sigmoid => Box::new(Sigmoid::default()),
    ActivationType::Softmax => Box::new(Softmax::default()),
  }
}