
use ganit::CVector;
use plexus::prelude::*;

fn main() {
  let l1 = DenseLayer::<6, 4>::new();
  let input_test: CVector<4, f64> = CVector::from([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
  ]);

  let outputs = l1.forward(input_test);
  println!("{:?}", outputs)
}