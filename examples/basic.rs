
use plexus::prelude::*;

fn main() {
  let l1 = DenseLayer::<6, 4>::new();
  let inputs = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
  ].into();

  let outputs = l1.forward(inputs);
  println!("{:?}", outputs)
}