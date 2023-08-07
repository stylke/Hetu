#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ArrayReshapeOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   NDArrayList& outputs, 
                                   RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

TensorList ArrayReshapeOpImpl::DoGradient(Operator& op, 
                                          const TensorList& grad_outputs) const {
  if (grad_outputs.at(0).is_defined() && grad_outputs.at(0))
    return {MakeArrayReshapeGradientOp(grad_outputs.at(0), op->input(0),
                                  op->grad_op_meta().set_name(op->grad_name()))};
  else 
    return { Tensor() };
}

HTShapeList ArrayReshapeOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  int64_t input_size = 1;
  HTShape input_shape = input_shapes.at(0);
  int64_t input_len = input_shape.size();
  // for (size_t i = 0; i < input_len; ++i) {
  //   input_size *= input_shape[i];
  // }
  // check if there exists -1 in output_shape
  int64_t idx = -1;
  size_t cnt = 0;
  int64_t output_size = 1;
  HTShape output_shape = get_output_shape();
  int64_t output_len = output_shape.size();
  for (size_t i = 0; i < input_len; ++i) {
    if (input_shape[i] == -1) {
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Input shape has more than one '-1' dims. ";
    }
    input_size *= input_shape[i];
  }
  cnt = 0;
  for (int64_t i = 0; i < output_len; ++i) {
    if (output_shape[i] == -1) {
      idx = i;
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Output shape has more than one '-1' dims. ";
    }
    output_size *= output_shape[i];
  }
  if (idx == -1) {
    HT_ASSERT(input_size == output_size) << "Invalid output size.";
  } else {
    output_size = output_size * (-1);
    HT_ASSERT(input_size % output_size == 0) << "Invalid output size." << input_shape << "," << output_shape
                                             << input_size << "," << output_size;
    output_shape[idx] = input_size / output_size;
  }
  return {output_shape};
}

void ArrayReshapeGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
ArrayReshapeGradientOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {input_shapes.at(1)};
}

Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          OpMeta op_meta) {
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeOpImpl>(output_shape),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

Tensor MakeArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input,
                                  OpMeta op_meta) {
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeGradientOpImpl>(),
      {std::move(grad_output), std::move(ori_input)},
      std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
