
// ===== Imports =====
use serde::{Serialize, Deserialize};
// ===================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetworkDescriptor {
  inputs: usize,
  hidden_layers: Vec<LayerDescriptor>,
  output_layer: OutputLayerDescriptor,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerDescriptor {
  neurons: usize,
  activation: ActivationType,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputLayerDescriptor {
  neurons: usize,
  loss: LossType,
  activation: ActivationType,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
  Step,
  Linear,
  ReLU,
  Sigmoid,
  Softmax,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
  CategoricalCrossEntropy,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
  Dense,
}