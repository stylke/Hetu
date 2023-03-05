#include "hetu/autograd/ops/ScalarLikeOp.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ScalarLikeOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, outputs.at(0),
                                  scalar_value(), stream());
}

TensorList ScalarLikeOpDef::DoGradient(const TensorList& grad_outputs) {
  return {Tensor()};
}

void ScalarLikeOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ScalarLikeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
