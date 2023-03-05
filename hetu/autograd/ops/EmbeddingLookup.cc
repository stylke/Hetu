#include "hetu/autograd/ops/EmbeddingLookup.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void EmbeddingLookupOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::EmbeddingLookup, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

TensorList EmbeddingLookupOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    EmbeddingLookupGradientOp(grad_outputs.at(0), _inputs[1], _outputs[0], _inputs[0],
                              grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

void EmbeddingLookupOpDef::DoInferMeta() {
  HTShape shape;
  if (_inputs[0]->has_shape() && _inputs[1]->has_shape()) {
    shape = _inputs[1]->shape();
    shape.emplace_back(_inputs[0]->shape(1));
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList
EmbeddingLookupOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape = input_shapes[1];
  output_shape.emplace_back(input_shapes[0][1]);
  return {output_shape};
}

void EmbeddingLookupGradientOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::EmbeddingLookupGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), stream());
}

void EmbeddingLookupGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[3]->meta());
}

HTShapeList
EmbeddingLookupGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  set_embed_shape(input_shapes.at(3));
  return {get_embed_shape()};
}

} // namespace autograd
} // namespace hetu
