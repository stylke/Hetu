#include "hetu/graph/ops/Sqrt.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SqrtOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Sqrt,
  //                                 inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
  NDArray::sqrt(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SqrtOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->require_grad(0) ? MakeMulByConstOp(MakeMulElewiseOp(MakeReciprocalSqrtOp(
                                          op->input(0), g_op_meta), grad_outputs.at(0), g_op_meta),
                                          0.5, g_op_meta.set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList SqrtOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes, 
                                     RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ReciprocalSqrtOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                     NDArrayList& outputs, 
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReciprocalSqrt, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList ReciprocalSqrtOpImpl::DoInferShape(Operator& op, 
                                               const HTShapeList& input_shapes, 
                                               RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeSqrtOp(Tensor input, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SqrtOpImpl>(),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeReciprocalSqrtOp(Tensor grad_output, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SqrtOpImpl>(),
    {std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
