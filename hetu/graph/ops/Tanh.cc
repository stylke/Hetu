#include "hetu/graph/ops/Tanh.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void TanhOpImpl::DoCompute(Operator& op, 
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  NDArray::tanh(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList TanhOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeTanhGradientOp(op->output(0), grad_outputs.at(0),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList TanhOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes, 
                                     RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void TanhGradientOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   NDArrayList& outputs, 
                                   RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::TanhGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList TanhGradientOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeTanhOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<TanhOpImpl>(),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeTanhGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<TanhGradientOpImpl>(),
    {std::move(input), std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
