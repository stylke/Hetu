#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SliceOpImpl::DoCompute(Operator& op, 
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Slice,
  //                                 inputs.at(0), outputs.at(0),
  //                                 get_begin_pos().data(), op->instantiation_ctx().stream());
  NDArray::slice(inputs.at(0), get_begin_pos(), outputs.at(0)->shape(),
                 op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SliceOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeSliceGradientOp(grad_outputs.at(0), op->output(0), op->input(0), get_begin_pos(),
                                get_output_shape(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList SliceOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes, 
                                      RuntimeContext& ctx) const {
  HTShape output_shape = get_output_shape();
  return {output_shape};
}

void SliceGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::SliceGradient, inputs.at(0),
    outputs.at(0), get_begin_pos().data(), op->instantiation_ctx().stream());
}


HTShapeList SliceGradientOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes, 
                                              RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

Tensor MakeSliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                   const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceOpImpl>(begin_pos, output_shape),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeSliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                           const HTShape& begin_pos, const HTShape& output_shape,
                           const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceGradientOpImpl>(begin_pos, output_shape),
    {std::move(grad_output), std::move(ori_output), std::move(ori_input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
