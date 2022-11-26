#include "hetu/autograd/ops/Pad.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void PadOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Pad,
                                  inputs.at(0), outputs.at(0), get_paddings(),
                                  stream(), get_mode(), get_constant());
}

TensorList PadOpDef::DoGradient(const TensorList& grad_outputs) {
  return {PadGradientOp(grad_outputs.at(0), get_paddings(), get_mode(),
                        grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList PadOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] + paddings[(i - (4 - len / 2)) * 2] +
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

void PadGradientOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::PadGradient, inputs.at(0),
    outputs.at(0), get_paddings(), stream(), get_mode());
}

HTShapeList PadGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] - paddings[(i - (4 - len / 2)) * 2] -
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

} // namespace autograd
} // namespace hetu
