#include "hetu/autograd/ops/Concat.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ConcatOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::Concat, inputs.at(0), inputs.at(1),
    outputs.at(0), get_axis(), stream());
}

TensorList ConcatOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_inputA =
    ConcatGradientOp(_inputs[0], grad_outputs.at(0), get_axis(), 0,
                     g_op_meta.set_name(grad_name(0)))
      ->output(0);
  auto grad_inputB =
    ConcatGradientOp(_inputs[1], grad_outputs.at(0), get_axis(), 1,
                     g_op_meta.set_name(grad_name(0)))
      ->output(0);
  return {grad_inputA, grad_inputB};
}

HTShapeList ConcatOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape shapeA = input_shapes.at(0);
  shapeA[get_axis()] += input_shapes.at(1)[get_axis()];
  return {shapeA};
}

void ConcatGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  if (placement().is_cuda()) {
    hetu::impl::ConcatGradientCuda(inputs.at(0), outputs.at(0), get_axis(),
                                   get_id(), stream());
  } else {
    hetu::impl::ConcatGradientCpu(inputs.at(0), outputs.at(0), get_axis(),
                                  get_id(), stream());
  }
}

HTShapeList ConcatGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
