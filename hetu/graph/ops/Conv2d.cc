#include "hetu/graph/ops/Conv2d.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void Conv2dOpImpl::DoCompute(Operator&op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2d,
  //                              inputs.at(0), inputs.at(1), outputs.at(0),
  //                              get_padding()[0], get_padding()[1],
  //                              get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
  NDArray::conv2d(inputs.at(0), inputs.at(1), get_padding(), get_stride(),
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList Conv2dOpImpl::DoGradient(Operator&op,
                                    const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->require_grad(0) ? MakeConv2dGradientofDataOp(
                                          op->input(1), grad_outputs.at(0), op->input(0), get_padding(),
                                          get_stride(), g_op_meta.set_name(op->grad_name(0)))
                                        : Tensor();
  auto grad_filter = op->require_grad(1) ? MakeConv2dGradientofFilterOp(op->input(0), grad_outputs.at(0), op->input(1),
                                           get_padding(), get_stride(),
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  return {grad_input, grad_filter};
}

HTShapeList Conv2dOpImpl::DoInferShape(Operator&op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dGradientofFilterOpImpl::DoCompute(Operator& op,
                                             const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dGradientofFilter,
    inputs.at(0), inputs.at(1), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
}

HTShapeList
Conv2dGradientofFilterOpImpl::DoInferShape(Operator&op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void Conv2dGradientofDataOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dGradientofData, inputs.at(0),
    inputs.at(1), outputs.at(0), get_padding()[0], get_padding()[1],
    get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
}

HTShapeList
Conv2dGradientofDataOpImpl::DoInferShape(Operator&op,
                                         const HTShapeList& input_shapes,
                                         RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void Conv2dAddBiasOpImpl::DoCompute(Operator& op,
                                    const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dAddBias, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
}

TensorList Conv2dAddBiasOpImpl::DoGradient(Operator& op,
                                           const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = MakeConv2dGradientofDataOp(
                      op->input(1), grad_outputs.at(0), op->input(0), get_padding(),
                      get_stride(), g_op_meta.set_name(op->grad_name(0)));
  auto grad_filter =
    MakeConv2dGradientofFilterOp(op->input(0), grad_outputs.at(0), op->input(1),
                             get_padding(), get_stride(),
                             g_op_meta.set_name(op->grad_name(1)));
  auto grad_bias = MakeReduceOp(grad_outputs.at(0), ReductionType::SUM, {0, 2, 3}, {false},
                                g_op_meta.set_name(op->grad_name(2)));
  return {grad_input, grad_filter, grad_bias};
}

HTShapeList Conv2dAddBiasOpImpl::DoInferShape(Operator& op,
                                              const HTShapeList& input_shapes,
                                              RuntimeContext& ctx) const {
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

Tensor MakeConv2dOp(Tensor input, Tensor filter, int64_t padding, int64_t stride,
                    const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dOpImpl>(padding, stride),
          {std::move(input), std::move(filter)},
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dGradientofFilterOp(Tensor input, Tensor grad_output, Tensor filter,
                                    const HTShape& padding, const HTStride& stride,
                                    const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dGradientofFilterOpImpl>(padding, stride),
          {std::move(input), std::move(grad_output), std::move(filter)},
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dGradientofDataOp(Tensor filter, Tensor grad_output, Tensor input,
                                  const HTShape& padding, const HTStride& stride,
                                  const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dGradientofDataOpImpl>(padding, stride),
          {std::move(filter), std::move(grad_output), std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dAddBiasOp(Tensor input, Tensor filter, Tensor bias, int64_t padding,
                           int64_t stride, const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dAddBiasOpImpl>(padding, stride),
          {std::move(input), std::move(filter), std::move(bias)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
