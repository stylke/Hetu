#include "hetu/graph/ops/LeakyRelu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LeakyReluOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  NDArray::leakyrelu(inputs.at(0), get_alpha(),
                     op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList LeakyReluOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeLeakyReluGradientOp(op->input(0), grad_outputs.at(0), get_alpha(),
                                op->grad_op_meta().set_name(op->grad_name(0)))
                              : Tensor()};
}

HTShapeList LeakyReluOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void LeakyReluGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::LeakyReluGradient, inputs.at(0),
    inputs.at(1), get_alpha(), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
LeakyReluGradientOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeLeakyReluOp(Tensor input, double alpha, const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<LeakyReluOpImpl>(alpha),
        {std::move(input)},
        std::move(op_meta))->output(0);   
}


Tensor MakeLeakyReluGradientOp(Tensor input, Tensor grad_output, double alpha,
                               const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<LeakyReluGradientOpImpl>(alpha),
          {std::move(input), std::move(grad_output)},
          std::move(op_meta))->output(0);   
}

} // namespace graph
} // namespace hetu
