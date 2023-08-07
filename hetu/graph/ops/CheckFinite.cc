#include "hetu/graph/ops/CheckFinite.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void CheckFiniteOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::CheckFinite,
                               inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList CheckFiniteOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList CheckFiniteOpImpl::DoInferShape(Operator& op, 
                                            const HTShapeList& input_shapes, 
                                            RuntimeContext& ctx) const {
  return {{1}};
}

Tensor MakeCheckFiniteOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
        std::make_shared<CheckFiniteOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
