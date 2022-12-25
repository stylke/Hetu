#include "hetu/autograd/ops/SoftmaxCrossEntropy.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace autograd {

using SCEOpDef = SoftmaxCrossEntropyOpDef;
using SCEGradOpDef = SoftmaxCrossEntropyGradientOpDef;

void SCEOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty_like(inputs.at(0));
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                                  hetu::impl::SoftmaxCrossEntropy, inputs.at(0),
                                  inputs.at(1), unreduced, stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false, stream_index(),
                    outputs.at(0));
  }
}

TensorList SCEOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = SoftmaxCrossEntropyGradientOp(
                      _inputs[0], _inputs[1], grad_outputs.at(0), reduction(),
                      grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList SCEOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else {
    HTShape output_shape = {};
    for (size_t i = 0; i < input_shapes.at(0).size() - 1; ++i) {
      output_shape.emplace_back(input_shapes.at(0)[i]);
    }
    return {output_shape};
  }
}

void SCEGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CUDA_ONLY(
      placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      broadcasted->numel(), broadcasted, HTAxes(), stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), stream());
  }
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::SoftmaxCrossEntropyGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), stream());
}

HTShapeList SCEGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
