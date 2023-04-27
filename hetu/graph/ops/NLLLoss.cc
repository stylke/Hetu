#include "hetu/graph/ops/NLLLoss.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

using NLLOpImpl = NLLLossOpImpl;
using NLLGradOpImpl = NLLLossGradientOpImpl;

void NLLOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) const {
  NDArray::nllloss(inputs.at(0), inputs.at(1), reduction(), 
                   op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList NLLOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeNLLLossGradientOp(
                                          op->input(0), op->input(1), grad_outputs.at(0), reduction(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList NLLOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else 
    return {input_shapes.at(1)};
}

void NLLGradOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0 / broadcasted->numel(), broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), op->instantiation_ctx().stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::NLLLossGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList NLLGradOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

Tensor MakeNLLLossOp(Tensor preds, Tensor labels,
                     ReductionType reduction,
                     OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<NLLLossOpImpl>(reduction),
        {std::move(preds), std::move(labels)},
        std::move(op_meta))->output(0);
}

Tensor MakeNLLLossOp(Tensor preds, Tensor labels,
                     const std::string& reduction ,
                     OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<NLLLossOpImpl>(Str2ReductionType(reduction)),
        {std::move(preds), std::move(labels)},
        std::move(op_meta))->output(0);
}

Tensor MakeNLLLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                             ReductionType reduction,
                             OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<NLLLossGradientOpImpl>(reduction),
        {std::move(preds), std::move(labels), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
