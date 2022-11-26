#include "hetu/autograd/ops/InstanceNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void InstanceNormOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HTShape local_shape = inputs.at(0)->shape();
  HT_ASSERT(local_shape.size() == 4);
  local_shape[3] = 1;
  local_shape[2] = 1;
  auto running_mean_meta = NDArrayMeta()
                             .set_device({kCUDA, 0})
                             .set_dtype(kFloat32)
                             .set_shape(local_shape);
  save_mean = NDArray(running_mean_meta);
  save_var = NDArray(running_mean_meta);
  set_shape(local_shape);
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::InstanceNorm, inputs.at(0),
    save_mean, save_var, outputs.at(0), get_eps(), stream());
}

TensorList InstanceNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto& self = reinterpret_cast<InstanceNormOp&>(get_self());
  auto grad_input = InstanceNormGradientOp(grad_outputs.at(0), _inputs[0], self,
                                           grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input};
}

HTShapeList InstanceNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void InstanceNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  InstanceNormOp forward_node = get_forward_node();
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::InstanceNormGradient, inputs.at(0),
    inputs.at(1), outputs.at(0), forward_node->save_mean,
    forward_node->save_var, forward_node->get_eps(), stream());
}

HTShapeList
InstanceNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
