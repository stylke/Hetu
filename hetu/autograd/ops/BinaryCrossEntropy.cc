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
