#include "hetu/graph/ops/Hardsigmoid.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void HardsigmoidOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::hardsigmoid(inputs.at(0),
                       op->instantiation_ctx().stream_index, 
                       outputs.at(0));
}

TensorList HardsigmoidOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeHardsigmoidGradientOp(op->output(0), grad_outputs.at(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void HardsigmoidGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::HardsigmoidGradient, inputs.at(0),
                               inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeHardsigmoidOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<HardsigmoidOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeHardsigmoidGradientOp(Tensor output, Tensor grad_output,
                                 OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<HardsigmoidGradientOpImpl>(),
        {std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
