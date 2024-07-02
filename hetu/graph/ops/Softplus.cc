#include "hetu/graph/ops/Softplus.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SoftplusOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::softplus(inputs.at(0), beta(), threshold(),
                    op->instantiation_ctx().stream_index, 
                    outputs.at(0));
}

TensorList SoftplusOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeSoftplusGradientOp(op->input(0), grad_outputs.at(0),
                                 beta(), threshold(), op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void SoftplusGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::SoftplusGradient, inputs.at(0),
                               inputs.at(1), beta(), threshold(),
                               outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeSoftplusOp(Tensor input, double beta, double threshold, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<SoftplusOpImpl>(beta, threshold),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeSoftplusGradientOp(Tensor input, Tensor grad_output,
                              double beta, double threshold, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<SoftplusGradientOpImpl>(beta, threshold),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
