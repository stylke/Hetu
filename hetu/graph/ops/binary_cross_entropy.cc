#include "hetu/graph/ops/binary_cross_entropy.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

TensorList
BinaryCrossEntropyOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  auto grad_probs =
    MakeBCEGradOp(op->input(0), op->input(1), grad_outputs.front(), reduction(),
                  op->grad_op_meta().set_name(op->grad_name()));
  return {grad_probs, Tensor()};
}

void BinaryCrossEntropyOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs,
                                         NDArrayList& outputs,
                                         RuntimeContext& runtime_ctx) const {
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty_like(inputs.at(0));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::BinaryCrossEntropy,
                                  inputs.at(0), inputs.at(1), unreduced,
                                  op->instantiation_ctx().stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false,
                    op->instantiation_ctx().stream_index, outputs.at(0));
  }
}

void BinaryCrossEntropyGradientOpImpl::DoCompute(
  Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
  RuntimeContext& runtime_ctx) const {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(),
      hetu::impl::BroadcastShapeMul, inputs.at(2), broadcasted->numel(),
      broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                    type(), hetu::impl::BroadcastShape,
                                    inputs.at(2), broadcasted, HTAxes(),
                                    op->instantiation_ctx().stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(),
    hetu::impl::BinaryCrossEntropyGradient, inputs.at(0), inputs.at(1),
    broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeBCEOp(Tensor probs, Tensor labels, ReductionType reduction,
                 OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<BinaryCrossEntropyOpImpl>(reduction),
                       {std::move(probs), std::move(labels)},
                       std::move(op_meta))
    ->output(0);
}

Tensor MakeBCEGradOp(Tensor probs, Tensor labels, Tensor grad_outputs,
                     ReductionType reduction, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<BinaryCrossEntropyGradientOpImpl>(reduction),
           {std::move(probs), std::move(labels), std::move(grad_outputs)},
           std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
