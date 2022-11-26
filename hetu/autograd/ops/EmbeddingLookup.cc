#include "hetu/autograd/ops/EmbeddingLookup.h"
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
    EmbeddingLookupGradientOp(grad_outputs.at(0), _inputs[1], _outputs[0],
                              grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList
EmbeddingLookupOpDef::DoInferShape(const HTShapeList& input_shapes) {
  set_grad_embed(input_shapes[0]);
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

HTShapeList
EmbeddingLookupGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  EmbeddingLookupOp& input_ptr =
    reinterpret_cast<EmbeddingLookupOp&>(_inputs[2]->producer());
  if (input_ptr) {
    set_embed_shape(input_ptr->get_grad_embed());
  }
  return {get_embed_shape()};
}

} // namespace autograd
} // namespace hetu
