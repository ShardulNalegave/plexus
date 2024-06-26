
pub mod iterator;

// ===== Imports =====
use std::ops::{
  Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign
};
use crate::{utils::Numeric, CVector, RVector};
use self::iterator::{MatrixEnumeratedIterator, MatrixIterator};
// ===================

/// # Matrix
/// Data-type to represent a Matrix of order `RxC` and containing values of type `T`.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<const R: usize, const C: usize, T: Numeric = f64> {
  data: [[T; C]; R],
}

impl<const R: usize, const C: usize, T: Numeric> From<[[T; C]; R]> for Matrix<R, C, T> {
  fn from(value: [[T; C]; R]) -> Self {
    assert!(R != 0, "Number of rows cannot be zero");
    assert!(C != 0, "Number of columns cannot be zero");

    Self { data: value }
  }
}

impl<const R: usize, const C: usize, T: Numeric> Into<[[T; C]; R]> for Matrix<R, C, T> {
  fn into(self) -> [[T; C]; R] {
    self.data
  }
}

impl<const R: usize, const C: usize, T: Numeric> Matrix<R, C, T> {
  /// # New
  /// Creates a new `Matrix` with the provided two-dimensional array.
  /// 
  /// **Note:** Both rows and columns should be >= 1.
  pub fn new(data: [[T; C]; R]) -> Self {
    assert!(R != 0, "Number of rows cannot be zero");
    assert!(C != 0, "Number of columns cannot be zero");

    Self { data }
  }

  /// # New with Fill
  /// Creates a new `Matrix` of the provided order and fills it with the provided value.
  /// 
  /// **Note:** Both rows and columns should be >= 1.
  pub fn new_with_fill(fill: T) -> Self {
    assert!(R != 0, "Number of rows cannot be zero");
    assert!(C != 0, "Number of columns cannot be zero");

    Self { data: [[fill; C]; R] }
  }

  /// # New with Generator
  /// Creates a new `Matrix` of the provided order and call the generator function to get each value.
  pub fn new_with_gen<F>(mut f: F) -> Self
    where
      F: FnMut((usize, usize)) -> T {
    assert!(R != 0, "Number of rows cannot be zero");
    assert!(C != 0, "Number of columns cannot be zero");
    
    Self {
      data: std::array::from_fn(|i| std::array::from_fn(|j| f((i, j)))),
    }
  }

  /// # Rows
  /// Returns the number of rows in matrix.
  pub fn rows(&self) -> usize { R }

  /// # Columns
  /// Returns the number of columns in matrix.
  pub fn cols(&self) -> usize { C }

  /// # Is Row Vector?
  /// Returns true if the matrix is a row vector, i.e. it has only one row.
  pub fn is_row_vector(&self) -> bool { R == 1 }

  /// # Is Column Vector?
  /// Returns true if the matrix is a column vector, i.e. it has only one column.
  pub fn is_col_vector(&self) -> bool { C == 1 }

  /// # Transpose
  /// Returns the transpose of matrix.
  /// ```
  /// use ganit::Matrix;
  /// 
  /// let matrix: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let transpose: Matrix<2, 2, i32> = [
  ///   [1, 3],
  ///   [2, 4],
  /// ].into();
  /// 
  /// assert!(matrix.transpose() == transpose);
  /// ```
  pub fn transpose(&self) -> Matrix<C, R, T> {
    let mut m = [[T::default(); R]; C];

    for (i, row) in self.data.iter().enumerate() {
      for (j, item) in row.iter().enumerate() {
        m[j][i] = *item;
      }
    }

    Matrix::new(m)
  }

  /// # Row
  /// Returns row `i` as a slice
  pub fn row(&self, i: usize) -> &[T; C] {
    &self.data[i]
  }

  /// # Add Row Vector
  /// Adds row vector to a matrix.
  /// ```
  /// use ganit::{Matrix, RVector};
  /// 
  /// let m1: Matrix<2, 3, i32> = [
  ///   [1, 2, 3],
  ///   [4, 5, 6],
  /// ].into();
  /// 
  /// let r1: RVector<3, i32> = [[1, 2, 3]].into();
  /// 
  /// let m2: Matrix<2, 3, i32> = [
  ///   [2, 4, 6],
  ///   [5, 7, 9],
  /// ].into();
  /// 
  /// assert!(m1.add_row_vector(r1) == m2);
  /// ```
  pub fn add_row_vector(&self, rhs: RVector<C, T>) -> Self {
    Matrix::new_with_gen(|(i, j)| self[(i, j)] + rhs[(0, j)])
  }

  /// # Add Column Vector
  /// Adds column vector to a matrix.
  /// ```
  /// use ganit::{Matrix, CVector};
  /// 
  /// let m1: Matrix<2, 3, i32> = [
  ///   [1, 2, 3],
  ///   [4, 5, 6],
  /// ].into();
  /// 
  /// let r1: CVector<2, i32> = [[1], [2]].into();
  /// 
  /// let m2: Matrix<2, 3, i32> = [
  ///   [2, 3, 4],
  ///   [6, 7, 8],
  /// ].into();
  /// 
  /// assert!(m1.add_column_vector(r1) == m2);
  /// ```
  pub fn add_column_vector(&self, rhs: CVector<R, T>) -> Self {
    Matrix::new_with_gen(|(i, j)| self[(i, j)] + rhs[(i, 0)])
  }

  pub fn div_row_vector(&self, rhs: RVector<R, T>) -> Self {
    Matrix::new_with_gen(|(i, j)| self[(i, j)] / rhs[(0, j)])
  }

  pub fn div_col_vector(&self, rhs: CVector<R, T>) -> Self {
    Matrix::new_with_gen(|(i, j)| self[(i, j)] / rhs[(i, 0)])
  }
}

/// Implementation for square matrices
impl<const N: usize, T: Numeric> Matrix<N, N, T> {
  /// # Determinant
  /// Returns the value of determinant of any square matrix.
  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// assert!(m.determinant() == -2);
  /// ```
  /// 
  /// Algorithm used to find determinant:-
  /// 1. Convert the matrix into a upper-triangle matrix by applying row transformations.
  /// 2. Determinant is equal to product of all values on the diagonal of matrix.
  pub fn determinant(&self) -> T {
    let mut m = self.data;

    for diagonal_index in 0..N {
      let diagonal = m[diagonal_index][diagonal_index];
      for i in (diagonal_index + 1)..N {
        let scaler = m[i][diagonal_index] / diagonal;
        for j in 0..N {
          m[i][j] = m[i][j] - (scaler * m[diagonal_index][j]);
        }
      }
    }

    let mut product: Option<T> = None;
    for i in 0..N {
      match product {
        None => product = Some(m[i][i]),
        Some(val) => product = Some(val * m[i][i]),
      }
    }

    product.unwrap()
  }
}

impl<const R: usize, const C: usize, T: Numeric> Index<(usize, usize)> for Matrix<R, C, T> {
  type Output = T;

  fn index(&self, index: (usize, usize)) -> &Self::Output {
    assert!(index.0 < R, "Row index out of bounds");
    assert!(index.1 < C, "Column index out of bounds");
    &self.data[index.0][index.1]
  }
}

impl<const R: usize, const C: usize, T: Numeric> IndexMut<(usize, usize)> for Matrix<R, C, T> {
  fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
    assert!(index.0 < R, "Row index out of bounds");
    assert!(index.1 < C, "Column index out of bounds");
    &mut self.data[index.0][index.1]
  }
}

impl<const R: usize, const C: usize, T: Numeric> Add for Matrix<R, C, T> {
  type Output = Self;

  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [5, 6],
  ///   [7, 8],
  /// ].into();
  /// 
  /// let m3: Matrix<2, 2, i32> = [
  ///   [6, 8],
  ///   [10, 12],
  /// ].into();
  /// 
  /// assert!((m1 + m2) == m3);
  /// ```
  fn add(self, rhs: Self) -> Self::Output {
    let mut lhs = self;

    for i in 0..lhs.rows() {
      for j in 0..lhs.cols() {
        lhs[(i, j)] += rhs[(i, j)];
      }
    }

    lhs
  }
}

impl<const R: usize, const C: usize, T: Numeric> AddAssign for Matrix<R, C, T> {
  /// ```
  /// use ganit::Matrix;
  /// 
  /// let mut m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [5, 6],
  ///   [7, 8],
  /// ].into();
  /// 
  /// let m3: Matrix<2, 2, i32> = [
  ///   [6, 8],
  ///   [10, 12],
  /// ].into();
  /// 
  /// m1 += m2;
  /// 
  /// assert!(m1 == m3);
  /// ```
  fn add_assign(&mut self, rhs: Self) {
    for i in 0..self.rows() {
      for j in 0..self.cols() {
        self[(i, j)] += rhs[(i, j)];
      }
    }
  }
}

impl<const R: usize, const C: usize, T: Numeric> Sub for Matrix<R, C, T> {
  type Output = Self;

  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [5, 6],
  ///   [7, 8],
  /// ].into();
  /// 
  /// let m3: Matrix<2, 2, i32> = [
  ///   [-4, -4],
  ///   [-4, -4],
  /// ].into();
  /// 
  /// assert!((m1 - m2) == m3);
  /// ```
  fn sub(self, rhs: Self) -> Self::Output {
    let mut lhs = self;

    for i in 0..lhs.rows() {
      for j in 0..lhs.cols() {
        lhs[(i, j)] -= rhs[(i, j)];
      }
    }

    lhs
  }
}

impl<const R: usize, const C: usize, T: Numeric> SubAssign for Matrix<R, C, T> {
  /// ```
  /// use ganit::Matrix;
  /// 
  /// let mut m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [5, 6],
  ///   [7, 8],
  /// ].into();
  /// 
  /// let m3: Matrix<2, 2, i32> = [
  ///   [-4, -4],
  ///   [-4, -4],
  /// ].into();
  /// 
  /// m1 -= m2;
  /// 
  /// assert!(m1 == m3);
  /// ```
  fn sub_assign(&mut self, rhs: Self) {
    for i in 0..self.rows() {
      for j in 0..self.cols() {
        self[(i, j)] -= rhs[(i, j)];
      }
    }
  }
}

impl<const R: usize, const C: usize, const P: usize, T: Numeric> Mul<Matrix<C, P, T>> for Matrix<R, C, T> {
  type Output = Matrix<R, P, T>;

  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [5, 6],
  ///   [7, 8],
  /// ].into();
  /// 
  /// let m3: Matrix<2, 2, i32> = [
  ///   [19, 22],
  ///   [43, 50],
  /// ].into();
  /// 
  /// assert!((m1 * m2) == m3);
  /// ```
  fn mul(self, rhs: Matrix<C, P, T>) -> Self::Output {
    let lhs = self;
    let mut data = [[T::default(); P]; R];

    for i in 0..lhs.rows() {
      for j in 0..rhs.cols() {
        for k in 0..lhs.cols() {
          data[i][j] += lhs[(i, k)] * rhs[(k, j)];
        }
      }
    }

    Matrix { data }
  }
}

impl<const R: usize, const C: usize, T: Numeric> Mul<T> for Matrix<R, C, T> {
  type Output = Self;

  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [4, 8],
  ///   [12, 16],
  /// ].into();
  /// 
  /// assert!(m1 * 4 == m2);
  /// ```
  fn mul(self, rhs: T) -> Self::Output {
    let mut lhs = self;
    for i in 0..lhs.rows() {
      for j in 0..lhs.cols() {
        lhs[(i, j)] *= rhs;
      }
    }

    lhs
  }
}

impl<const R: usize, const C: usize, T: Numeric> PartialEq for Matrix<R, C, T> {
  /// ```
  /// use ganit::Matrix;
  /// 
  /// let m1: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// let m2: Matrix<2, 2, i32> = [
  ///   [1, 2],
  ///   [3, 4],
  /// ].into();
  /// 
  /// assert!(m1 == m2);
  /// ```
  fn eq(&self, other: &Self) -> bool {
    self.data
      .iter()
      .zip(other.data.iter())
      .all(|(a, b)| {
        a.iter().zip(b.iter())
          .all(|(c, d)| c == d)
      })
  }
}

impl<const R: usize, const C: usize, T: Numeric + Copy> FromIterator<((usize, usize), T)> for Matrix<R, C, T> {
  fn from_iter<I: IntoIterator<Item = ((usize, usize), T)>>(iter: I) -> Self {
    let mut data = [[T::default(); C]; R];

    for (index, item) in iter {
      data[index.0][index.1] = item;
    }

    Matrix { data }
  }
}

impl<const R: usize, const C: usize, T: Numeric + Copy> FromIterator<T> for Matrix<R, C, T> {
  fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
    let mut data = [[T::default(); C]; R];
    let mut row = 0;
    let mut col = 0;

    for item in iter {
      data[row][col] = item;
      col += 1;

      if col >= C {
        col = 0;
        row += 1;
      }
    }

    Matrix { data }
  }
}

impl<'a, const R: usize, const C: usize, T: Numeric> Matrix<R, C, T> {
  pub fn into_iter(&'a self) -> MatrixIterator<'a, R, C, T> {
    MatrixIterator::new(self)
  }

  pub fn into_enumerated_iter(&'a self) -> MatrixEnumeratedIterator<'a, R, C, T> {
    MatrixEnumeratedIterator::new(self)
  }
}