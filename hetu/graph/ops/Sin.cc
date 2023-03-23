#include "hetu/graph/ops/Sin.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SinOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Sin,
  //                                 inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
  NDArray::sin(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SinOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeCosOp(op->input(0), 
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList SinOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void CosOpImpl::DoCompute(Operator& op,
                          const NDArrayList& inputs, NDArrayList& outputs, 
                          RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Cos, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList CosOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeSinOp(Tensor input, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SinOpImpl>(),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeCosOp(Tensor input, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<CosOpImpl>(),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
