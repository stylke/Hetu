#include "hetu/graph/ops/Pad.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void PadOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Pad,
  //                                 inputs.at(0), outputs.at(0), get_paddings(),
  //                                 op->instantiation_ctx().stream(), get_mode(), get_constant());
  NDArray::pad(inputs.at(0), get_paddings(), get_mode(), get_constant(),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList PadOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakePadGradientOp(grad_outputs.at(0), get_paddings(), get_mode(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList PadOpImpl::DoInferShape(Operator& op, 
                                    const HTShapeList& input_shapes, 
                                    RuntimeContext& ctx) const {
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] + paddings[(i - (4 - len / 2)) * 2] +
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

void PadGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::PadGradient, inputs.at(0),
    outputs.at(0), get_paddings(), op->instantiation_ctx().stream(), get_mode());
}

HTShapeList PadGradientOpImpl::DoInferShape(Operator& op, 
                                            const HTShapeList& input_shapes, 
                                            RuntimeContext& ctx) const {
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] - paddings[(i - (4 - len / 2)) * 2] -
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

Tensor MakePadOp(Tensor input, const HTShape& paddings, std::string mode, double constant,
                 const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<PadOpImpl>(paddings, mode, constant),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

Tensor MakePadGradientOp(Tensor grad_output, const HTShape& paddings, std::string mode,
                         const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<PadGradientOpImpl>(paddings, mode),
        {std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
