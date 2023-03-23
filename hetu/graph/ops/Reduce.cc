#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/Broadcast.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ReduceOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  // if (reduction() == ReductionType::MEAN) {
  //   HT_DISPATCH_KERNEL_CUDA_ONLY(
  //     op->instantiation_ctx().placement.type(), type(), hetu::impl::ReduceMean, inputs.at(0),
  //     outputs.at(0), get_axes().data(), get_axes().size(), op->instantiation_ctx().stream());
  // } else if (reduction() == ReductionType::SUM) {
  //   HT_DISPATCH_KERNEL_CUDA_ONLY(
  //     op->instantiation_ctx().placement.type(), type(), hetu::impl::ReduceSum, inputs.at(0),
  //     outputs.at(0), get_axes().data(), get_axes().size(), op->instantiation_ctx().stream());
  // }
  NDArray::reduce(inputs.at(0), reduction(), get_axes(), false,
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ReduceOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeReduceGradientOp(grad_outputs.at(0), op->output(0), op->input(0), HTShape(), reduction(),
                                get_axes(), get_keepdims(), op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList ReduceOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  HTShapeList outputlist = {};
  HTShape input_shape = input_shapes.at(0);
  int ndim = input_shape.size();
  int64_t mean_multiplier = 1;
  HTShape axes = get_axes();
  int len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  HTShape add_axes = {};
  for (int i = 0; i < len; ++i) {
    if (axes[i] < 0) {
      axes[i] += ndim;
    }
    HT_ASSERT(axes[i] >= 0 && axes[i] < ndim);
    mean_multiplier *= input_shape[axes[i]];
    if (keepdims[i] == true)
      input_shape[axes[i]] = 1;
    else {
      input_shape[axes[i]] = 0;
      add_axes.emplace_back(axes[i]);
    }
  }
  HTShape output_shape(0);
  for (int i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  outputlist = {output_shape};
  return outputlist;
}

void  ReduceGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs, RuntimeContext& ctx) const {
  if (reduction() == ReductionType::MEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(0),
      get_const_value(), outputs.at(0), get_add_axes(), op->instantiation_ctx().stream());
  } else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), get_add_axes(), op->instantiation_ctx().stream());
  }
}

HTShapeList ReduceGradientOpImpl::DoInferShape(Operator& op,
                                               const HTShapeList& input_shapes,
                                               RuntimeContext& ctx) const {
  return  {input_shapes.at(2)};
}

Tensor MakeReduceOp(Tensor input, ReductionType reduction, const HTAxes& axes,
                    const HTKeepDims& keepdims,
                    const OpMeta& op_meta) {
  HTAxes parsed_axes = axes;
  HTKeepDims parsed_keepdims = keepdims;
  if (parsed_axes.size() == 0) {
      parsed_axes.reserve(input->ndim());
      for (size_t i = 0; i < input->ndim(); ++i) {
        parsed_axes.push_back(i);
      }
    }
  parsed_axes = NDArrayMeta::ParseAxes(parsed_axes, input->ndim());
  HT_ASSERT(parsed_keepdims.size() == parsed_axes.size() || parsed_keepdims.size() == 1);
  if (parsed_keepdims.size() == 1) {
    int len = parsed_axes.size();
    bool keepdim = parsed_keepdims[0];
    for (int i = 1; i < len; ++i) {
      parsed_keepdims.emplace_back(keepdim);
    }
  }     
  return Graph::MakeOp(
          std::make_shared<ReduceOpImpl>(reduction, parsed_axes, parsed_keepdims),
          {std::move(input)},
          std::move(op_meta))->output(0);              
}

Tensor MakeReduceOp(Tensor input, const std::string& mode, const HTAxes& axes,
                    const HTKeepDims& keepdims,
                    const OpMeta& op_meta) {
  HTAxes parsed_axes = axes;
  HTKeepDims parsed_keepdims = keepdims;
  if (parsed_axes.size() == 0) {
      parsed_axes.reserve(input->ndim());
      for (size_t i = 0; i < input->ndim(); ++i) {
        parsed_axes.push_back(i);
      }
    }
  parsed_axes = NDArrayMeta::ParseAxes(parsed_axes, input->ndim());
  HT_ASSERT(parsed_keepdims.size() == parsed_axes.size() || parsed_keepdims.size() == 1);
  if (parsed_keepdims.size() == 1) {
    int len = parsed_axes.size();
    bool keepdim = parsed_keepdims[0];
    for (int i = 1; i < len; ++i) {
      parsed_keepdims.emplace_back(keepdim);
    }
  }     
  return Graph::MakeOp(
          std::make_shared<ReduceOpImpl>(Str2ReductionType(mode), parsed_axes, parsed_keepdims),
          {std::move(input)},
          std::move(op_meta))->output(0);       
}

Tensor MakeReduceGradientOp(Tensor input, Tensor ori_output, Tensor ori_input, const HTShape& shape,
                            ReductionType reduction, const HTAxes add_axes, const HTKeepDims& keepdims,
                            const OpMeta& op_meta){
  double const_value = 0;
  if (reduction == ReductionType::MEAN) {
    HTShape input_shape = ori_input->shape();
    int ndim = input_shape.size();
    int64_t mean_multiplier = 1;
    int len = add_axes.size();
    for (int i = 0; i < len; ++i) {
      HT_ASSERT(add_axes[i] >= 0 && add_axes[i] < ndim);
      mean_multiplier *= input_shape[add_axes[i]];
    }
    const_value = 1.0 / mean_multiplier;
  }
  return Graph::MakeOp(
          std::make_shared<ReduceGradientOpImpl>(shape, reduction, add_axes, keepdims, const_value),
          {std::move(input), std::move(ori_output), std::move(ori_input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
