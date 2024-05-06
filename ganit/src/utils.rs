
// ===== Imports =====
use std::ops::{
  Add, AddAssign,
  Div, DivAssign,
  Mul, MulAssign,
  Rem, RemAssign,
  Sub, SubAssign,
};
// ===================

pub trait Numeric: 
  Default +
  Sized + Copy + Clone +
  Add + AddAssign +
  Sub + SubAssign +
  Mul + MulAssign +
  Div + DivAssign +
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