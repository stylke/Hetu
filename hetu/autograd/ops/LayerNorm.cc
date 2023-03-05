#include "hetu/autograd/ops/LayerNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void LayerNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                               hetu::impl::LayerNorm, inputs.at(0),
                               inputs.at(1), inputs.at(2), outputs.at(1), outputs.at(2),
                               outputs.at(0), get_eps(), stream());
}

TensorList LayerNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = LayerNormGradientOp(grad_outputs.at(0), _inputs[0],
                                        _inputs[1], _outputs[1], _outputs[2], get_eps(), g_op_meta);
  return {grad_input->output(0), grad_input->output(1), grad_input->output(2)};
}

void LayerNormOpDef::DoInferMeta() {
  HTShape local_shape = _inputs[0]->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  AddOutput(_inputs[0]->meta());
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device()).set_dtype(_inputs[0]->dtype()).set_shape(local_shape));
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device()).set_dtype(_inputs[0]->dtype()).set_shape(local_shape));
}

HTShapeList LayerNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape local_shape = input_shapes.at(0);
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  return {input_shapes.at(0), local_shape, local_shape};
}

void LayerNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  int64_t channels = inputs.at(0)->shape(1);
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::LayerNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), inputs.at(3), inputs.at(4),
    get_eps(), stream());
}

void LayerNormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
  AddOutput(_inputs[2]->meta());
  AddOutput(_inputs[2]->meta());
}

HTShapeList
LayerNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(2)};
}

} // namespace autograd
} // namespace hetu
