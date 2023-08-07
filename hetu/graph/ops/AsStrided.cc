#include "hetu/graph/ops/AsStrided.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AsStridedOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::AsStrided, inputs.at(0),
                               outputs.at(0), get_stride(), op->instantiation_ctx().stream());
}

TensorList AsStridedOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeAsStridedGradientOp(grad_outputs.at(0), op->input(0), get_stride(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
AsStridedOpImpl::DoInferShape(Operator& op,
                              const HTShapeList& input_shapes,
                              RuntimeContext& ctx) const {
  return {outshape()};
}

void AsStridedOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "AsStridedOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

void AsStridedGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::AsStridedGradient,
    inputs.at(0), outputs.at(0), get_stride(), op->instantiation_ctx().stream());
}

HTShapeList
AsStridedGradientOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {input_shapes[1]};
}

void AsStridedGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

Tensor MakeAsStridedOp(Tensor input, HTShape outshape, HTShape stride, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedOpImpl>(outshape, stride),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAsStridedGradientOp(Tensor grad_output, Tensor input, HTShape stride,
                               OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedGradientOpImpl>(stride),
           {std::move(grad_output), std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
