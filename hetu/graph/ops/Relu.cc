#include "hetu/graph/ops/Relu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ReluOpImpl::DoCompute(Operator& op, 
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Relu,
  //                                 inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
  NDArray::relu(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ReluOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeReluGradientOp(op->input(0), grad_outputs.at(0),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList ReluOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes, 
                                     RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ReluGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReluGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList ReluGradientOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeReluOp(Tensor input, const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<ReluOpImpl>(),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

Tensor MakeReluGradientOp(Tensor input, Tensor grad_output,
                          const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<ReluGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
