#include "hetu/autograd/ops/SoftmaxCrossEntropy.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

using SCEOpDef = SoftmaxCrossEntropyOpDef;
using SCEGradOpDef = SoftmaxCrossEntropyGradientOpDef;

void SCEOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                               hetu::impl::SoftmaxCrossEntropy, inputs.at(0),
                               inputs.at(1), outputs.at(0), stream());
}

TensorList SCEOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    SoftmaxCrossEntropyGradientOp(_inputs[0], _inputs[1], grad_outputs.at(0),
                                  grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList SCEOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  HTShape output_shape = {};
  for (size_t i = 0; i < input_shapes.at(0).size() - 1; ++i) {
    output_shape.emplace_back(input_shapes.at(0)[i]);
  }
  return {output_shape};
}

void SCEOpDef::DeduceStates() {
  DistributedStates ds_preds = _inputs[0]->get_distributed_states();
  DistributedStates ds_labels = _inputs[1]->get_distributed_states();
  int ndim = _inputs[0]->ndim();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid())
    << "SoftmaxCrossEntropyOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  HT_ASSERT(ds_preds.check_max_dim(ndim - 1)) // cannot split in last dimension
    << "Input tensor can only support split in dimension < " << ndim - 1;
  _outputs[0]->set_distributed_states(ds_preds);
}

void SCEGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  if (placement().is_cuda()) {
    hetu::impl::SoftmaxCrossEntropyGradientCuda(
      inputs.at(0), inputs.at(1), inputs.at(2), outputs.at(0), stream());
  }
}

HTShapeList SCEGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void SCEGradOpDef::DeduceStates() {
  DistributedStates ds_preds = _inputs[0]->get_distributed_states();
  DistributedStates ds_labels = _inputs[1]->get_distributed_states();
  DistributedStates ds_grad_output = _inputs[2]->get_distributed_states();
  int ndim = _inputs[0]->ndim();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid() && ds_grad_output.is_valid())
    << "SoftmaxCrossEntropyOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels) && ds_labels.check_equal(ds_grad_output))
    << "Distributed states among preds and labels and grad_output should be equal!";
  HT_ASSERT(ds_preds.check_max_dim(ndim - 1)) // cannot split in last dimension
    << "Input tensor can only support split in dimension < " << ndim - 1;
  _outputs[0]->set_distributed_states(ds_preds);
}

} // namespace autograd
} // namespace hetu
