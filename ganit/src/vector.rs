
// ===== Imports =====
use crate::matrix::Matrix;
// ===================

/// # Row Vector
/// Type shorthand for representing a Row-Vector.
/// Vectors are nothing but one-dimensional Matrices, a Row-Vector is a matrix of order `1xN`
pub type RVector<const N: usize, T> = Matrix<1, N, T>;

/// # Column Vector
/// Type shorthand for representing a Column-Vector.
/// Vectors are nothing but one-dimensional Matrices, a Column-Vector is a matrix of order `Nx1`
pub type CVector<const N: usize, T> = Matrix<N, 1, T>;