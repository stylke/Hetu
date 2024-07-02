#include "hetu/graph/ops/Hardtanh.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void HardtanhOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::hardtanh(inputs.at(0), min_val(), max_val(),
                    op->instantiation_ctx().stream_index, 
                    outputs.at(0));
}

TensorList HardtanhOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeHardtanhGradientOp(op->output(0), grad_outputs.at(0),
                                 min_val(), max_val(), op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void HardtanhGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::HardtanhGradient, inputs.at(0),
                               inputs.at(1), min_val(), max_val(),
                               outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeHardtanhOp(Tensor input, double min_val, double max_val, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<HardtanhOpImpl>(min_val, max_val),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeHardtanhGradientOp(Tensor output, Tensor grad_output,
                              double min_val, double max_val, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<HardtanhGradientOpImpl>(min_val, max_val),
        {std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
