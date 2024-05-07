
/// Utilities
pub mod utils;

/// Provides data-type for representing Matrices
pub mod matrix;

/// Provides type shorthands for representing Vectors
pub mod vector;

pub use crate::{
  utils::Numeric,
  matrix::Matrix,
  vector::{RVector, CVector},
};