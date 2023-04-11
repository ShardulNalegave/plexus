
// ===== Imports =====
use serde::{Serialize, Deserialize};
// ===================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetworkDescriptor {
  inputs: usize,
  loss: LossType,
  layers: Vec<LayerDescriptor>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerDescriptor {
  neurons: usize,
  activation: ActivationType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
  Step,
  Linear,
  ReLU,
  Sigmoid,
  Softmax,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
  CategoricalCrossEntropy
}