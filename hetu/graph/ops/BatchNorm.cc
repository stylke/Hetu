#include "hetu/graph/ops/BatchNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void BatchNormOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  // TODO: Convert these states to VariableOps
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                                 hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(3)), 0,
  //                                 op->instantiation_ctx().stream());

  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(
  //   op->instantiation_ctx().placement.type(), type(), hetu::impl::ArraySet, 
  //   const_cast<NDArray&>(inputs.at(4)), 1, op->instantiation_ctx().stream());
  NDArray::arrayset(const_cast<NDArray&>(inputs.at(3)), 0,
                    op->instantiation_ctx().stream_index);
  NDArray::arrayset(const_cast<NDArray&>(inputs.at(4)), 1,
                    op->instantiation_ctx().stream_index);
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(
  //   op->instantiation_ctx().placement.type(), type(), hetu::impl::BatchNorm, inputs.at(0),
  //   inputs.at(1), inputs.at(2), outputs.at(0), get_momentum(), get_eps(),
  //   const_cast<NDArray&>(inputs.at(3)), const_cast<NDArray&>(inputs.at(4)), 
  //   outputs.at(1), outputs.at(2), op->instantiation_ctx().stream());
  NDArray::batchnorm(inputs.at(0), inputs.at(1), inputs.at(2), 
                     const_cast<NDArray&>(inputs.at(3)),
                     const_cast<NDArray&>(inputs.at(4)),
                     get_momentum(), get_eps(),
                     op->instantiation_ctx().stream_index,
                     outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList BatchNormOpImpl::DoGradient(Operator& op,
                                       const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  auto grad = op->requires_grad(0) ? MakeBatchNormGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                     op->output(1), op->output(2), get_eps(), g_op_meta)
                                   : empty;                         
  return {grad.at(0), grad.at(1), grad.at(2), Tensor(), Tensor()};
}

HTShapeList BatchNormOpImpl::DoInferShape(Operator& op,
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(0).size() == 4);
  return {input_shapes.at(0), {input_shapes.at(0)[1]}, {input_shapes.at(0)[1]}};
}

void BatchNormGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::BatchNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), get_eps(), const_cast<NDArray&>(inputs.at(3)),
    const_cast<NDArray&>(inputs.at(4)), op->instantiation_ctx().stream());
}

HTShapeList
BatchNormGradientOpImpl::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  int64_t channels = input_shapes.at(0)[1];
  return {input_shapes.at(1), {channels}, {channels}};
}

TensorList MakeBatchNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
                       Tensor running_mean, Tensor running_var,
                       double momentum, double eps,
                       OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<BatchNormOpImpl>(momentum, eps),
        {std::move(input), std::move(bn_scale), std::move(bn_bias),
         std::move(running_mean), std::move(running_var)},
        std::move(op_meta))->outputs();
}

TensorList MakeBatchNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                               Tensor save_mean, Tensor save_var, double eps,
                               OpMeta op_meta) {
  return Graph::MakeOp(
                 std::make_shared<BatchNormGradientOpImpl>(eps),
                 {std::move(output_grad), std::move(input),
                 std::move(bn_scale), std::move(save_mean),
                 std::move(save_var)},
                 std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
