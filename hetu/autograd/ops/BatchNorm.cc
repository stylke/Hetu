#include "hetu/autograd/ops/BatchNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BatchNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  // TODO: Convert these states to VariableOps
  int64_t channels = inputs.at(0)->shape(1);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(3)), 0,
                                  stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(4)), 1, stream());
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::BatchNorm, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_momentum(), get_eps(),
    const_cast<NDArray&>(inputs.at(3)), const_cast<NDArray&>(inputs.at(4)), 
    outputs.at(1), outputs.at(2), stream());
}

TensorList BatchNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad = BatchNormGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                  _outputs[1], _outputs[2], get_eps(), g_op_meta);                         
  return {grad->output(0), grad->output(1), grad->output(2), Tensor(), Tensor()};
}

void BatchNormOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
  int64_t channels = _inputs[0]->shape(1);
  HTShape shape = {channels};
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device())
                         .set_dtype(_inputs[0]->dtype()).set_shape(shape));
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device())
                         .set_dtype(_inputs[0]->dtype()).set_shape(shape));
}

HTShapeList BatchNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT(input_shapes.at(0).size() == 4);
  return {input_shapes.at(0), {input_shapes.at(0)[1]}, {input_shapes.at(0)[1]}};
}

void BatchNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::BatchNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), get_eps(), const_cast<NDArray&>(inputs.at(3)),
    const_cast<NDArray&>(inputs.at(4)), stream());
}

void BatchNormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
  AddOutput(_inputs[2]->meta());
  AddOutput(_inputs[2]->meta());
}

HTShapeList
BatchNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t channels = input_shapes.at(0)[1];
  return {input_shapes.at(1), {channels}, {channels}};
}

} // namespace autograd
} // namespace hetu
