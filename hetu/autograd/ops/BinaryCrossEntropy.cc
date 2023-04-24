#include "hetu/autograd/ops/BinaryCrossEntropy.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

using BCEOpDef = BinaryCrossEntropyOpDef;
using BCEGradOpDef = BinaryCrossEntropyGradientOpDef;

void BCEOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::BinaryCrossEntropy, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

TensorList BCEOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    BinaryCrossEntropyGradientOp(_inputs[0], _inputs[1], grad_outputs.at(0),
                                 grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList BCEOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void BCEOpDef::DeduceStates() {
  DistributedStates ds_preds = _inputs[0]->get_distributed_states();
  DistributedStates ds_labels = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid()
            && ds_preds.get_device_num() == ds_labels.get_device_num())
    << "BCEOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Inputs tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  // HT_ASSERT(ds_preds.check_max_dim(1))
  //   << "BCEOp only support data parallel!";
  _outputs[0]->set_distributed_states(ds_preds);  
}

void BCEGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  if (placement().is_cuda()) {
    hetu::impl::BinaryCrossEntropyGradientCuda(
      inputs.at(0), inputs.at(1), inputs.at(2), outputs.at(0), stream());
  } else {
    hetu::impl::BinaryCrossEntropyGradientCpu(
      inputs.at(0), inputs.at(1), inputs.at(2), outputs.at(0), stream());
  }
}

HTShapeList BCEGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
