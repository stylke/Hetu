#include "hetu/autograd/ops/AvgPool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void AvgPoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::AvgPool,
                               inputs.at(0), get_kernel_H(), get_kernel_W(),
                               outputs.at(0), get_padding(), get_stride(),
                               stream());
}

TensorList AvgPoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {AvgPoolGradientOp(_outputs[0], grad_outputs.at(0), _inputs[0],
                            get_kernel_H(), get_kernel_W(), get_padding(),
                            get_stride(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList AvgPoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void AvgPoolGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::AvgPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), stream());
}

HTShapeList
AvgPoolGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

} // namespace autograd
} // namespace hetu
