#include "hetu/graph/ops/Roll.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void RollOpImpl::DoCompute(Operator& op,
                           const NDArrayList& inputs,
                           NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                              hetu::impl::Roll, inputs.at(0),
  //                              shifts(), dims(),
  //                              outputs.at(0), op->instantiation_ctx().stream());
  NDArray::roll(inputs.at(0), shifts(), dims(),
                op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList RollOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HTShape negshifts = shifts();
  for (auto &bit: negshifts) {
    bit = - bit;
  }
  auto grad_input = op->require_grad(0) ? MakeRollOp(grad_outputs.at(0), negshifts, dims(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
RollOpImpl::DoInferShape(Operator& op, 
                         const HTShapeList& input_shapes, 
                         RuntimeContext& ctx) const {
  return {input_shapes[0]};
}

Tensor MakeRollOp(Tensor input, HTShape shifts,
                  HTAxes dims, const OpMeta& op_meta) {
  return Graph::MakeOp(
      std::make_shared<RollOpImpl>(shifts, dims),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
