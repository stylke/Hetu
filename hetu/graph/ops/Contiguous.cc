#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList ContiguousOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        RuntimeContext& ctx) const {
  NDArrayList outputs = inputs.at(0)->is_contiguous() ? inputs : DoAllocOutputs(op, inputs, ctx);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Contiguous, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
  return outputs;
}

TensorList ContiguousOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeContiguousGradientOp(grad_outputs.at(0), op->input(0)->stride(),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

HTShapeList ContiguousOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes, 
                                          RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeContiguousOp(Tensor input, OpMeta op_meta) {
  auto contig_op = Graph::MakeOp(
    std::make_shared<ContiguousOpImpl>(),
    {std::move(input)},
    std::move(op_meta));
  input->set_contiguous_op_id(contig_op->id());
  return contig_op->output(0);
}

void ContiguousGradientOpImpl::DoCompute(Operator& op, 
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {                             
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ContiguousGradient, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList ContiguousGradientOpImpl::DoInferShape(Operator& op, 
                                                   const HTShapeList& input_shapes, 
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeContiguousGradientOp(Tensor input, const HTStride& stride, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<ContiguousGradientOpImpl>(stride),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
