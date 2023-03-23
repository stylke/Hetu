#include "hetu/graph/ops/Softmax.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SoftmaxOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::Softmax,
  //                              inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
  NDArray::softmax(inputs.at(0), get_dim(), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SoftmaxOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeSoftmaxGradientOp(op->output(0), grad_outputs.at(0),
                                get_dim(), op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList SoftmaxOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void SoftmaxGradientOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::SoftmaxGradient, inputs.at(0),
                               inputs.at(1), outputs.at(0),
                               get_dim(), op->instantiation_ctx().stream());
}

HTShapeList
SoftmaxGradientOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeSoftmaxOp(Tensor input, int64_t dim, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxOpImpl>(dim),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeSoftmaxGradientOp(Tensor input, Tensor grad_output,
                             int64_t dim, const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxGradientOpImpl>(dim),
    {std::move(input), std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
