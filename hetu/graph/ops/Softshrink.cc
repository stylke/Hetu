#include "hetu/graph/ops/Softshrink.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SoftshrinkOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::softshrink(inputs.at(0), lambda(),
                      op->instantiation_ctx().stream_index, 
                      outputs.at(0));
}

TensorList SoftshrinkOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeSoftshrinkGradientOp(op->output(0), grad_outputs.at(0),
                                 lambda(), op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void SoftshrinkGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::SoftshrinkGradient, inputs.at(0),
                               inputs.at(1), lambda(),
                               outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeSoftshrinkOp(Tensor input, double lambda, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<SoftshrinkOpImpl>(lambda),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeSoftshrinkGradientOp(Tensor output, Tensor grad_output,
                                double lambda, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<SoftshrinkGradientOpImpl>(lambda),
        {std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
