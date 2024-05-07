
// ===== Imports =====
use std::ops::{
  Add, AddAssign,
  Div, DivAssign,
  Mul, MulAssign,
  Rem, RemAssign,
  Sub, SubAssign,
};
// ===================

/// # Numeric
/// Marker trait for numerical types in Rust.
pub trait Numeric: 
  Default +
  Sized + Copy + Clone +
  PartialEq +
  Add<Output = Self> + AddAssign +
  Sub<Output = Self> + SubAssign +
  Mul<Output = Self> + MulAssign +
  Div<Output = Self> + DivAssign +
  Rem + RemAssign {}

impl Numeric for isize {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}

impl Numeric for usize {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}

impl Numeric for f32 {}
impl Numeric for f64 {}