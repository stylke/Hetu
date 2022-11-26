#include "hetu/autograd/ops/Softmax.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SoftmaxOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::Softmax,
                               inputs.at(0), outputs.at(0), stream());
}

TensorList SoftmaxOpDef::DoGradient(const TensorList& grad_outputs) {
  return {SoftmaxGradientOp(_outputs[0], grad_outputs.at(0),
                            grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList SoftmaxOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void SoftmaxGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                               hetu::impl::SoftmaxGradient, inputs.at(0),
                               inputs.at(1), outputs.at(0), stream());
}

HTShapeList
SoftmaxGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
