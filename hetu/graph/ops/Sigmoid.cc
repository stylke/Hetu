#include "hetu/graph/ops/Sigmoid.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SigmoidOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                                 hetu::impl::Sigmoid, inputs.at(0),
  //                                 outputs.at(0), op->instantiation_ctx().stream());
  NDArray::sigmoid(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SigmoidOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->require_grad(0) ? MakeMulElewiseOp(MakeMulElewiseOp(op->output(0),
                                          MakeAddByConstOp(MakeNegateOp(op->output(0), g_op_meta), 1, g_op_meta),
                                          g_op_meta), grad_outputs.at(0), g_op_meta.set_name(op->grad_name(0)))
                                        : Tensor();
  return {grad_input};
}

HTShapeList SigmoidOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeSigmoidOp(Tensor input, const OpMeta& op_meta) {
    return Graph::MakeOp(
      std::make_shared<SigmoidOpImpl>(),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
