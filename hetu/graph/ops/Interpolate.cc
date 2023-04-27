#include "hetu/graph/ops/Interpolate.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void InterpolateOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Interpolate, inputs.at(0),
                                  outputs.at(0), align_corners(), op->instantiation_ctx().stream());
}

TensorList InterpolateOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeInterpolateGradientOp(grad_outputs.at(0), op->input(0), align_corners(), scale_factor(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
InterpolateOpImpl::DoInferShape(Operator& op, 
                                const HTShapeList& input_shapes, 
                                RuntimeContext& ctx) const {
  HTShape output = input_shapes[0];
  if (out_shape().size() == 2) {
    output[2] = out_shape()[0];
    output[3] = out_shape()[1];
  }
  else {
    HT_ASSERT(scale_factor() > 0);
    output[2] = output[2] * scale_factor();
    output[3] = output[3] * scale_factor();
  }
  return {output};
}

void InterpolateGradientOpImpl::DoCompute(Operator& op,
                                          const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::InterpolateGradient,
    inputs.at(0), outputs.at(0), align_corners(), op->instantiation_ctx().stream());
}


HTShapeList
InterpolateGradientOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(1)};
}

Tensor MakeInterpolateOp(Tensor input, const HTShape& outshape,
                         bool align_corners, double scale_factor,
                         OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<InterpolateOpImpl>(outshape, align_corners, scale_factor),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeInterpolateGradientOp(Tensor grad_output, Tensor input,
                                 bool align_corners, double scale_factor,
                                 OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<InterpolateGradientOpImpl>(align_corners, scale_factor),
          {std::move(grad_output), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
