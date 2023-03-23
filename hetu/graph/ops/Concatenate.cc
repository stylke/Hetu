#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ConcatenateOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  // int num = inputs.size();
  // size_t offset = 0;
  // size_t axis = get_axis();
  // for (int i = 0; i < num; ++i) {
  //   HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                                   hetu::impl::Concatenate, inputs.at(i),
  //                                   outputs.at(0), axis, offset, op->instantiation_ctx().stream());
  //   offset += inputs.at(i)->shape(axis);
  // }
  NDArray::cat(inputs, get_axis(), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ConcatenateOpImpl::DoGradient(Operator& op,
                                         const TensorList& grad_outputs) const {
  TensorList grads = {};
  int num = op->num_inputs();
  int outdim = 0; 
  size_t axis = get_axis();
  auto g_op_meta = op->grad_op_meta();
  for (int i = 0; i < num; ++i) {
    HT_ASSERT(op->input(i)->shape(axis) >= 0);
    auto grad_input = op->require_grad(i) ? MakeConcatenateGradientOp(
                                            op->input(i), op->output(0), grad_outputs.at(0), 
                                            axis, outdim, g_op_meta.set_name(op->grad_name(i)))
                                          : Tensor();
    outdim += op->input(i)->shape(axis);
    grads.emplace_back(grad_input);
  }
  return grads;
}

HTShapeList ConcatenateOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& ctx) const {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  int ind = 0;
  ind += 1;
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        ind += 1;
        out_dim += shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

void ConcatenateGradientOpImpl::DoCompute(Operator& op,
                                          const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::ConcatenateGradient, inputs.at(2),
    outputs.at(0), get_axis(), get_offset(), op->instantiation_ctx().stream());
}

HTShapeList
ConcatenateGradientOpImpl::DoInferShape(Operator& op,
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeConcatenateOp(const TensorList& inputs, size_t axis,
                         const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatenateOpImpl>(axis),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

Tensor MakeConcatenateGradientOp(Tensor input, Tensor output, Tensor grad_output, size_t axis, size_t offset,
                                 const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatenateGradientOpImpl>(axis, offset),
          {std::move(input), std::move(output), std::move(grad_output)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
