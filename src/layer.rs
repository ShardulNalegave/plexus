
mod dense;

pub use self::{
  dense::Dense,
};

// ===== Imports =====
use crate::descriptor::LayerDescriptor;
// ===================

pub trait Layer {
  fn forward(&mut self, inputs: &na::DMatrix<f64>) -> na::DMatrix<f64>;
  fn backward(&mut self, gradients: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}
