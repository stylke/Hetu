#include "hetu/graph/ops/Elu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void EluOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::elu(inputs.at(0), alpha(), scale(), 
               op->instantiation_ctx().stream_index, 
               outputs.at(0));
}

TensorList EluOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeEluGradientOp(op->output(0), grad_outputs.at(0),
                                 alpha(), scale(), op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void EluGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::EluGradient, inputs.at(0),
                               inputs.at(1), alpha(), scale(),
                               outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeEluOp(Tensor input, double alpha, double scale, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<EluOpImpl>(alpha, scale),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeEluGradientOp(Tensor output, Tensor grad_output,
                         double alpha, double scale, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<EluGradientOpImpl>(alpha, scale),
        {std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
