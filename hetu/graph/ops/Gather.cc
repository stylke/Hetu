#include "hetu/graph/ops/Gather.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void GatherOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs,
                             NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::gather(inputs.at(0), inputs.at(1), get_dim(), 
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList GatherOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  auto grad_input = op->require_grad(0) ? MakeGatherGradientOp(grad_outputs.at(0), get_dim(), op->input(1), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList
GatherOpImpl::DoInferShape(Operator& op,
                           const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes[0].size() == input_shapes[1].size());
  int64_t len = input_shapes[0].size();
  for (int64_t i = 0; i < len; ++i) {
    if (i != get_dim())
      HT_ASSERT(input_shapes[0][i] == input_shapes[1][i]);
  }
  return {input_shapes.at(1)};
}

void GatherGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::GatherGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), get_dim(), op->instantiation_ctx().stream());
}

HTShapeList
GatherGradientOpImpl::DoInferShape(Operator& op,
                                   const HTShapeList& input_shapes,
                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

Tensor MakeGatherOp(Tensor input, int64_t dim, Tensor id, const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<GatherOpImpl>(dim),
          {std::move(input), std::move(id)},
          std::move(op_meta))->output(0);
}

Tensor MakeGatherGradientOp(Tensor grad_output, int64_t dim, Tensor id, Tensor input,
                            const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<GatherGradientOpImpl>(dim),
          {std::move(grad_output), std::move(id), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
