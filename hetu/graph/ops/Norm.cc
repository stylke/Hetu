#include "hetu/graph/ops/Norm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void NormOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::Norm,
  //                              inputs.at(0), outputs.at(0), dim(), getp(), op->instantiation_ctx().stream());
  NDArray::norm(inputs.at(0), getp(), dim(), keepdim(),
                op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList NormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeNormGradientOp(op->input(0), op->output(0), grad_outputs.at(0), getp(), dim(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList NormOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HTShape outshape = input_shapes.at(0);
  int64_t axi = dim() >= 0 ? dim(): dim() + outshape.size();
  if (keepdim()) 
    outshape[axi] = 1;
  else 
    outshape.erase(outshape.begin() + axi);
  return {outshape};
}

void NormGradientOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::NormGradient, inputs.at(0),
                               inputs.at(1), inputs.at(2), outputs.at(0), dim(), getp(), 
                               op->instantiation_ctx().stream());
}

HTShapeList NormGradientOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeNormOp(Tensor input, int64_t p, int64_t dim, 
                  bool keepdim, const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<NormOpImpl>(p, dim, keepdim),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

Tensor MakeNormGradientOp(Tensor input, Tensor output, Tensor grad_output, int64_t p, 
                          int64_t dim, const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<NormGradientOpImpl>(p, dim),
        {std::move(input), std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
