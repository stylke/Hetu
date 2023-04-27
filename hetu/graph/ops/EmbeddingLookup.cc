#include "hetu/graph/ops/EmbeddingLookup.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void EmbeddingLookupOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  NDArray::embedding(inputs.at(0), inputs.at(1), 
                     op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList EmbeddingLookupOpImpl::DoGradient(Operator& op,
                                             const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeEmbeddingLookupGradientOp(grad_outputs.at(0), op->input(1), op->output(0), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList
EmbeddingLookupOpImpl::DoInferShape(Operator& op,
                                    const HTShapeList& input_shapes,
                                    RuntimeContext& ctx) const {
  HTShape output_shape = input_shapes[1];
  output_shape.emplace_back(input_shapes[0][1]);
  return {output_shape};
}

void EmbeddingLookupGradientOpImpl::DoCompute(Operator& op,
                                              const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::EmbeddingLookupGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
EmbeddingLookupGradientOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext &ctx) const {
  return {input_shapes.at(3)};
}

Tensor MakeEmbeddingLookupOp(Tensor input, Tensor id, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupOpImpl>(),
          {std::move(input), std::move(id)},
          std::move(op_meta))->output(0);
}

Tensor MakeEmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input, Tensor input,
                                     OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupGradientOpImpl>(),
          {std::move(grad_output), std::move(id), std::move(ori_input), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
