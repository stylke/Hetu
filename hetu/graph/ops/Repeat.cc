#include "hetu/graph/ops/Repeat.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void RepeatOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
  //                              hetu::impl::Repeat, inputs.at(0),
  //                              outputs.at(0), op->instantiation_ctx().stream());
  NDArray::repeat(inputs.at(0), repeats(),
                  op->instantiation_ctx().stream_index,
                  outputs.at(0));
}

TensorList RepeatOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->require_grad(0) ? MakeRepeatGradientOp(grad_outputs.at(0), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
RepeatOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HTShape output_shape = repeats();
  HT_ASSERT(output_shape.size() >= input_shapes[0].size());
  for (size_t i = 0; i < input_shapes[0].size(); ++i) {
    output_shape[i + output_shape.size() - input_shapes[0].size()] *= input_shapes[0][i]; 
  }
  return {output_shape};
}

void RepeatGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs, NDArrayList& outputs,
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::RepeatGradient,
    inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
RepeatGradientOpImpl::DoInferShape(Operator& op, 
                                   const HTShapeList& input_shapes, 
                                   RuntimeContext& ctx) const {
  return {input_shapes[1]};
}

Tensor MakeRepeatOp(Tensor input, HTShape repeats, const OpMeta& op_meta) {
    return Graph::MakeOp(
        std::make_shared<RepeatOpImpl>(repeats),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

Tensor MakeRepeatGradientOp(Tensor grad_output, Tensor input,
                            const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<RepeatGradientOpImpl>(),
        {std::move(grad_output), std::move(input)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
