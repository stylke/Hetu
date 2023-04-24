#include "hetu/autograd/ops/BatchNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BatchNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  // TODO: Convert these states to VariableOps
  int64_t channels = inputs.at(0)->shape(1);
  auto running_mean_meta = NDArrayMeta()
                             .set_device({kCUDA, 0})
                             .set_dtype(kFloat32)
                             .set_shape({channels});
  running_mean = NDArray(running_mean_meta);
  running_var = NDArray(running_mean_meta);
  save_mean = NDArray(running_mean_meta);
  save_var = NDArray(running_mean_meta);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, running_mean, 0,
                                  stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ArraySet, running_var, 1, stream());
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::BatchNorm, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_momentum(), get_eps(),
    running_mean, running_var, save_mean, save_var, stream());
}

TensorList BatchNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto& self = reinterpret_cast<BatchNormOp&>(get_self());
  auto g_op_meta = grad_op_meta();
  auto grad = BatchNormGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                  self, get_eps(), g_op_meta)
                ->output(0);
  auto grad_data = BatchNormGradientofDataOp(grad, _inputs[0],
                                             g_op_meta.set_name(grad_name(0)))
                     ->output(0);
  auto grad_scale = BatchNormGradientofScaleOp(grad, _inputs[1],
                                               g_op_meta.set_name(grad_name(1)))
                      ->output(0);
  auto grad_bias = BatchNormGradientofBiasOp(grad, _inputs[2],
                                             g_op_meta.set_name(grad_name(2)))
                     ->output(0);
  return {grad_data, grad_scale, grad_bias};
}

HTShapeList BatchNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT(input_shapes.at(0).size() == 4);
  return {input_shapes.at(0)};
}

// 注: input tensor shape=[N, C, H, W], 在N, H, W维上做切分均会影响到batch norm的mean和var, 
// 导致最终结果产生差异(类比于batch和mini-batch做batchnorm的区别)
void BatchNormOpDef::DeduceStates() {
  auto ds_input = _inputs[0]->get_distributed_states();
  auto ds_scale = _inputs[1]->get_distributed_states();
  auto ds_bias = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << name() << ": input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2)) // cannot split in H,W dimension
    << "Input tensor can only support split in dimension N, C!";
  HT_ASSERT(ds_input.get_dim(1) == ds_scale.get_dim(0) && ds_input.get_dim(1) == ds_bias.get_dim(0))
    << "Split states for bn_scale and bn_bias should be equal to split states for input dimension C!";  
  _outputs[0]->set_distributed_states(ds_input);
}

void BatchNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  BatchNormOp forward_node = get_forward_node();
  int64_t channels = inputs.at(0)->shape(1);
  auto running_mean_meta = NDArrayMeta()
                             .set_device({kCUDA, 0})
                             .set_dtype(kFloat32)
                             .set_shape({channels});
  tmp_gradient_bn_arr = NDArray(running_mean_meta);
  tmp_gradient_bn_scale = NDArray(running_mean_meta);
  tmp_gradient_bn_bias = NDArray(running_mean_meta);
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::BatchNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), tmp_gradient_bn_arr, tmp_gradient_bn_scale,
    tmp_gradient_bn_bias, get_eps(), forward_node->save_mean,
    forward_node->save_var, stream());
}

HTShapeList
BatchNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void BatchNormGradientOpDef::DeduceStates() {
  DistributedStates ds_output_grad = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_output_grad.is_valid())
    << "BatchNormGradientOpDef: distributed states for output_grad tensor must be valid!";
  HT_ASSERT(ds_output_grad.get_dim(-2) == 1)
    << "Output_grad tensor shouldn't be partial!";
  _outputs[0]->set_distributed_states(ds_output_grad);
}

void BatchNormGradientofDataOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {}

HTShapeList
BatchNormGradientofDataOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void BatchNormGradientofDataOpDef::DeduceStates() {
  DistributedStates ds_data = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_data.is_valid())
    << "BatchNormGradientofDataOpDef: distributed states for data tensor must be valid!";
  HT_ASSERT(ds_data.get_dim(-2) == 1)
    << "Data tensor shouldn't be partial!";
  _outputs[0]->set_distributed_states(ds_data);  
}

void BatchNormGradientofScaleOpDef::DoCompute(const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& ctx) {}

HTShapeList
BatchNormGradientofScaleOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void BatchNormGradientofScaleOpDef::DeduceStates() {
  DistributedStates ds_scale = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_scale.is_valid())
    << "BatchNormGradientofScaleOpDef: distributed states for scale tensor must be valid!";
  HT_ASSERT(ds_scale.get_dim(-2) == 1)
    << "Scale tensor shouldn't be partial!";
  _outputs[0]->set_distributed_states(ds_scale);  
}

void BatchNormGradientofBiasOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {}

HTShapeList
BatchNormGradientofBiasOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void BatchNormGradientofBiasOpDef::DeduceStates() {
  DistributedStates ds_bias = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_bias.is_valid())
    << "BatchNormGradientofBiasOpDef: distributed states for bias tensor must be valid!";
  HT_ASSERT(ds_bias.get_dim(-2) == 1)
    << "Bias tensor shouldn't be partial!";
  _outputs[0]->set_distributed_states(ds_bias);  
}

} // namespace autograd
} // namespace hetu
