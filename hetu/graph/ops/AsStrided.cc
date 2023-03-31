#include "hetu/graph/ops/AsStrided.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AsStridedOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::AsStrided, inputs.at(0),
                               outputs.at(0), get_stride(), op->instantiation_ctx().stream());
}

TensorList AsStridedOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->require_grad(0) ? MakeAsStridedGradientOp(grad_outputs.at(0), op->input(0), get_stride(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
AsStridedOpImpl::DoInferShape(Operator& op,
                              const HTShapeList& input_shapes,
                              RuntimeContext& ctx) const {
  return {outshape()};
}

void AsStridedGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::AsStridedGradient,
    inputs.at(0), outputs.at(0), get_stride(), op->instantiation_ctx().stream());
}

HTShapeList
AsStridedGradientOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {input_shapes[1]};
}

Tensor MakeAsStridedOp(Tensor input, HTShape outshape, HTShape stride, const OpMeta& op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedOpImpl>(outshape, stride),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAsStridedGradientOp(Tensor grad_output, Tensor input, HTShape stride,
                               const OpMeta& op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedGradientOpImpl>(stride),
           {std::move(grad_output), std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace autograd
} // namespace hetu
