
// ===== Imports =====
use crate::{Matrix, Numeric};
// ===================

pub struct MatrixIterator<'a, const R: usize, const C: usize, T: Numeric> {
  matrix: &'a Matrix<R, C, T>,
  row: usize,
  col: usize,
}

impl<'a, const R: usize, const C: usize, T: Numeric> MatrixIterator<'a, R, C, T> {
  pub fn new(matrix: &'a Matrix<R, C, T>) -> Self {
    Self { matrix, row: 0, col: 0 }
  }
}

impl<'a, const R: usize, const C: usize, T: Numeric> Iterator for MatrixIterator<'a, R, C, T> {
  type Item = T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R {
      return None;
    }

    let item = self.matrix[(self.row, self.col)];

    self.col += 1;
    if self.col >= C {
      self.col = 0;
      self.row += 1;
    }

    Some(item)
  }
}

pub struct MatrixEnumeratedIterator<'a, const R: usize, const C: usize, T: Numeric> {
  matrix: &'a Matrix<R, C, T>,
  row: usize,
  col: usize,
}

impl<'a, const R: usize, const C: usize, T: Numeric> MatrixEnumeratedIterator<'a, R, C, T> {
  pub fn new(matrix: &'a Matrix<R, C, T>) -> Self {
    Self { matrix, row: 0, col: 0 }
  }
}

impl<'a, const R: usize, const C: usize, T: Numeric> Iterator for MatrixEnumeratedIterator<'a, R, C, T> {
  type Item = ((usize, usize), T);

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R {
      return None;
    }

    let index = (self.row, self.col);
    let item = self.matrix[index];

    self.col += 1;
    if self.col >= C {
      self.col = 0;
      self.row += 1;
    }

    Some((index, item))
  }
}
