#include "hetu/graph/ops/SoftmaxCrossEntropy.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

using SCEOpImpl = SoftmaxCrossEntropyOpImpl;
using SCEGradOpImpl = SoftmaxCrossEntropyGradientOpImpl;

void SCEOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::sceloss(inputs.at(0), inputs.at(1), reduction(),
                   op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SCEOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->require_grad(0) ? MakeSoftmaxCrossEntropyGradientOp(
                                          op->input(0), op->input(1), grad_outputs.at(0), reduction(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList SCEOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else {
    HTShape output_shape = {};
    for (size_t i = 0; i < input_shapes.at(0).size() - 1; ++i) {
      output_shape.emplace_back(input_shapes.at(0)[i]);
    }
    return {output_shape};
  }
}

void SCEGradOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  HTShape output_shape = HTShape(inputs.at(0)->shape().begin(), inputs.at(0)->shape().end() - 1);
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty(output_shape, 
                                           inputs.at(0)->device(), inputs.at(0)->dtype());
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0f / broadcasted->numel(), broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), op->instantiation_ctx().stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::SoftmaxCrossEntropyGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList SCEGradOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 ReductionType reduction,
                                 const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropyOpImpl>(reduction),
    {std::move(preds), std::move(labels)},
    std::move(op_meta))->output(0);
}

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 const std::string& reduction,
                                 const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropyOpImpl>(Str2ReductionType(reduction)),
    {std::move(preds), std::move(labels)},
    std::move(op_meta))->output(0);
}

Tensor MakeSoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                         ReductionType reduction,
                                         const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropyGradientOpImpl>(reduction),
    {std::move(preds), std::move(labels), std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
