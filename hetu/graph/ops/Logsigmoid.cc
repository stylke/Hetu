#include "hetu/graph/ops/Logsigmoid.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LogsigmoidOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::logsigmoid(inputs.at(0), false,
                      op->instantiation_ctx().stream_index, 
                      outputs.at(0));
}

TensorList LogsigmoidOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeLogsigmoidGradientOp(op->input(0), grad_outputs.at(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void LogsigmoidGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::LogsigmoidGradient, inputs.at(0),
                               inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeLogsigmoidOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<LogsigmoidOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeLogsigmoidGradientOp(Tensor input, Tensor grad_output,
                               OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<LogsigmoidGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
