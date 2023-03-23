#include "hetu/graph/ops/InstanceNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void InstanceNormOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(
  //   op->instantiation_ctx().placement.type(), type(), hetu::impl::InstanceNorm, inputs.at(0),
  //   outputs.at(1), outputs.at(2), outputs.at(0), get_eps(), op->instantiation_ctx().stream());
  NDArray::instancenorm(inputs.at(0), get_eps(), op->instantiation_ctx().stream_index,
                        outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList InstanceNormOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto grad_input = op->require_grad(0) ? MakeInstanceNormGradientOp(grad_outputs.at(0), op->input(0), op->output(1), op->output(2), get_eps(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList InstanceNormOpImpl::DoInferShape(Operator&op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  HTShape local_shape = input_shapes.at(0);
  local_shape[3] = 1;
  local_shape[2] = 1;
  return {input_shapes.at(0), local_shape, local_shape};
}

void InstanceNormGradientOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::InstanceNormGradient, inputs.at(0),
    inputs.at(1), outputs.at(0), const_cast<NDArray&>(inputs.at(2)),
    const_cast<NDArray&>(inputs.at(3)), get_eps(), op->instantiation_ctx().stream());
}

HTShapeList
InstanceNormGradientOpImpl::DoInferShape(Operator& op,
                                         const HTShapeList& input_shapes,
                                         RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

TensorList MakeInstanceNormOp(Tensor input, double eps,
                              const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<InstanceNormOpImpl>(eps),
          {std::move(input)},
          std::move(op_meta))->outputs();                                
}

Tensor MakeInstanceNormGradientOp(Tensor output_grad, Tensor input,
                                  Tensor save_mean, Tensor save_var,
                                  double eps,
                                  const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<InstanceNormGradientOpImpl>(eps),
          {std::move(output_grad), std::move(input), std::move(save_mean), std::move(save_var)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
