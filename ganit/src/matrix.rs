
use std::ops::{
  Index, IndexMut,
  Add, AddAssign, 
  Sub, SubAssign,
};

// ===== Imports =====
use crate::utils::Numeric;
// ===================

pub struct Matrix<T: Numeric = isize> {
  rows: usize,
  cols: usize,
  data: Vec<Vec<T>>,
}

impl<T: Numeric> Matrix<T> {
  pub fn new(data: Vec<Vec<T>>) -> Self {
    assert_ne!(data.len(), 0);
    assert_ne!(data[0].len(), 0);

    let rows = data.len();
    let cols = data[0].len();

    Self {
      data,
      rows,
      cols,
    }
  }

  pub fn new_with_fill(rows: usize, cols: usize, fill_with: T) -> Self {
    Self {
      data: vec![vec![fill_with; cols]; rows],
      rows, cols,
    }
  }

  pub fn rows(&self) -> usize {
    self.rows
  }

  pub fn cols(&self) -> usize {
    self.cols
  }

  pub fn transpose(&self) -> Matrix<T> {
    let mut m: Vec<Vec<T>> = vec![vec![T::default(); self.rows]; self.cols];

    for (i, row) in self.data.iter().enumerate() {
      for (j, item) in row.iter().enumerate() {
        m[j][i] = *item;
      }
    }

    Matrix::new(m)
  }
}

impl<T: Numeric> Index<(usize, usize)> for Matrix<T> {
  type Output = T;

  fn index(&self, index: (usize, usize)) -> &Self::Output {
    &self.data[index.0][index.1]
  }
}

impl<T: Numeric> IndexMut<(usize, usize)> for Matrix<T> {
  fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
    &mut self.data[index.0][index.1]
  }
}

impl<T: Numeric> Add for Matrix<T> {
  type Output = Self;

  fn add(mut self, rhs: Self) -> Self::Output {
    assert_eq!(self.rows, rhs.rows);
    assert_eq!(self.cols, rhs.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self[(i, j)] += rhs[(i, j)]
      }
    }

    self
  }
}

impl<T: Numeric> AddAssign for Matrix<T> {
  fn add_assign(&mut self, rhs: Self) {
    assert_eq!(self.rows, rhs.rows);
    assert_eq!(self.cols, rhs.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self[(i, j)] += rhs[(i, j)]
      }
    }
  }
}

impl<T: Numeric> Sub for Matrix<T> {
  type Output = Self;

  fn sub(mut self, rhs: Self) -> Self::Output {
    assert_eq!(self.rows, rhs.rows);
    assert_eq!(self.cols, rhs.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self[(i, j)] -= rhs[(i, j)]
      }
    }

    self
  }
}

impl<T: Numeric> SubAssign for Matrix<T> {
  fn sub_assign(&mut self, rhs: Self) {
    assert_eq!(self.rows, rhs.rows);
    assert_eq!(self.cols, rhs.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self[(i, j)] -= rhs[(i, j)]
      }
    }
  }
}
