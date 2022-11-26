#include "hetu/autograd/ops/LayerNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void LayerNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HTShape local_shape = inputs.at(0)->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  auto running_mean_meta = NDArrayMeta()
                             .set_device({kCUDA, 0})
                             .set_dtype(kFloat32)
                             .set_shape(local_shape);
  save_mean = NDArray(running_mean_meta);
  save_var = NDArray(running_mean_meta);
  set_shape(local_shape);
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                               hetu::impl::LayerNorm, inputs.at(0),
                               inputs.at(1), inputs.at(2), save_mean, save_var,
                               outputs.at(0), get_eps(), stream());
}

TensorList LayerNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto& self = reinterpret_cast<LayerNormOp&>(get_self());
  auto grad_input = LayerNormGradientOp(grad_outputs.at(0), _inputs[0],
                                        _inputs[1], self, get_eps(), g_op_meta)
                      ->output(0);
  auto data_gradient =
    LayerNormGradientofDataOp(grad_input, _inputs[0],
                              g_op_meta.set_name(grad_name(0)))
      ->output(0);
  auto scale_gradient =
    LayerNormGradientofScaleOp(grad_input, _inputs[1],
                               g_op_meta.set_name(grad_name(1)))
      ->output(0);
  auto bias_gradient =
    LayerNormGradientofBiasOp(grad_input, _inputs[2],
                              g_op_meta.set_name(grad_name(2)))
      ->output(0);
  return {data_gradient, scale_gradient, bias_gradient};
}

HTShapeList LayerNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void LayerNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  LayerNormOp forward_node = get_forward_node();
  int64_t channels = inputs.at(0)->shape(1);
  auto running_mean_meta = NDArrayMeta()
                             .set_device({kCUDA, 0})
                             .set_dtype(kFloat32)
                             .set_shape({channels});
  tmp_gradient_bn_arr = NDArray(running_mean_meta);
  tmp_gradient_bn_scale = NDArray(running_mean_meta);
  tmp_gradient_bn_bias = NDArray(running_mean_meta);
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::LayerNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), tmp_gradient_bn_arr, tmp_gradient_bn_scale,
    tmp_gradient_bn_bias, forward_node->save_mean, forward_node->save_var,
    get_eps(), stream());
}

HTShapeList
LayerNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void LayerNormGradientofDataOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {}

HTShapeList
LayerNormGradientofDataOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void LayerNormGradientofScaleOpDef::DoCompute(const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& ctx) {}

HTShapeList
LayerNormGradientofScaleOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void LayerNormGradientofBiasOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {}

HTShapeList
LayerNormGradientofBiasOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

} // namespace autograd
} // namespace hetu
