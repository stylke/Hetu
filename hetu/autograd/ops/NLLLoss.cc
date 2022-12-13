#include "hetu/autograd/ops/NLLLoss.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace autograd {

using NLLOpDef = NLLLossOpDef;
using NLLGradOpDef = NLLLossGradientOpDef;

void NLLOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  NDArray tmp = NDArray::empty(inputs.at(1)->shape(), inputs.at(0)->device(), inputs.at(0)->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::NLLLoss, inputs.at(0),
                                  inputs.at(1), tmp, stream());
  if (reduce()) {
    HTAxes reduce_axes(tmp->ndim());
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    if (reduction() == "mean") {
      HT_DISPATCH_KERNEL_CUDA_ONLY(
        placement().type(), type(), hetu::impl::ReduceMean, tmp,
        outputs.at(0), reduce_axes.data(), reduce_axes.size(), stream());
    }
    else if (reduction() == "sum") {
      HT_DISPATCH_KERNEL_CUDA_ONLY(
        placement().type(), type(), hetu::impl::ReduceMean, tmp,
        outputs.at(0), reduce_axes.data(), reduce_axes.size(), stream());
    }
    else {
      HT_NOT_IMPLEMENTED << "invalid reduction type:" << reduction();
    }
  }
}

TensorList NLLOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    NLLLossGradientOp(_inputs[0], _inputs[1], grad_outputs.at(0), reduce(), reduction(),
                      grad_op_meta().set_name(grad_name()))->output(0);
  return {grad_input, Tensor()};
}

HTShapeList NLLOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  // return {input_shapes.at(0)};
  if (reduce())
    return {{1}};
  else 
    return {input_shapes.at(1)};
}

void NLLGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  if (reduce()) {
    NDArray tmp = NDArray::empty_like(inputs.at(0));
    HTAxes reduce_axes(tmp->ndim());
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    if (reduction() == "mean") {
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(
          placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
          tmp->numel(), tmp, reduce_axes, stream());
    }
    else if (reduction() == "sum") {
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(
          placement().type(), type(), hetu::impl::BroadcastShape, inputs.at(2),
          tmp, reduce_axes, stream());
    }                     
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::NLLLossGradient, inputs.at(0), inputs.at(1), 
                                    tmp, outputs.at(0), stream());
  }
  else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::NLLLossGradient, inputs.at(0), inputs.at(1), 
                                    inputs.at(2), outputs.at(0), stream());
  }
}

HTShapeList NLLGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
