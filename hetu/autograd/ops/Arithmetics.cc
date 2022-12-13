#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace autograd {

void AddElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::add(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList AddElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0), grad_outputs.at(0)};
}

HTShapeList AddElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void AddByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::add(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList AddByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0)};
}

HTShapeList AddByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void SubElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::sub(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList SubElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0),
          NegateOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList SubElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void SubByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::sub(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList SubByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0)};
}

HTShapeList SubByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void SubFromConstOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  NDArray::sub(const_value(), inputs.at(0), stream_index(), outputs.at(0));
}

TensorList SubFromConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {NegateOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList SubFromConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void NegateOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  NDArray::neg(inputs.at(0), stream_index(), outputs.at(0));
}

TensorList NegateOpDef::DoGradient(const TensorList& grad_outputs) {
  return {NegateOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList NegateOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void MulElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::mul(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList MulElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_a = MulElewiseOp(grad_outputs.at(0), _inputs[1],
                             g_op_meta.set_name(grad_name(0)))
                  ->output(0);
  auto grad_b = MulElewiseOp(grad_outputs.at(0), _inputs[0],
                             g_op_meta.set_name(grad_name(1)))
                  ->output(0);
  return {grad_a, grad_b};
}

HTShapeList MulElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void MulByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::mul(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList MulByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {MulByConstOp(grad_outputs.at(0), const_value(),
                       grad_op_meta().set_name(grad_name(0)))
            ->output(0)};
}

HTShapeList MulByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void DivElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::div(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList DivElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  // 1 / b
  auto dividend_grad = ReciprocalOp(_inputs.at(1), g_op_meta)->output(0);
  auto grad_a = MulElewiseOp(dividend_grad, grad_outputs.at(0),
                             g_op_meta.set_name(grad_name(0)))
                  ->output(0);
  // - a / (b^2) = - (a / b) / b
  auto divisor_grad =
    NegateOp(DivElewiseOp(_outputs[0], _inputs.at(1), g_op_meta)->output(0),
             g_op_meta)
      ->output(0);
  auto grad_b = MulElewiseOp(divisor_grad, grad_outputs.at(0),
                             g_op_meta.set_name(grad_name(1)))
                  ->output(0);
  return {grad_a, grad_b};
}

HTShapeList DivElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void DivByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::div(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList DivByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {DivByConstOp(grad_outputs.at(0), const_value(),
                       grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList DivByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void DivFromConstOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  NDArray::div(const_value(), inputs.at(0), stream_index(), outputs.at(0));
}

TensorList DivFromConstOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  // - c / (x^2) = - (c / x) / x
  auto divisor_grad =
    NegateOp(DivElewiseOp(_outputs[0], _inputs.at(1), g_op_meta)->output(0),
             g_op_meta)
      ->output(0);
  auto grad_input = MulElewiseOp(divisor_grad, grad_outputs.at(0),
                                 g_op_meta.set_name(grad_name(1)))
                      ->output(0);
  return {grad_input};
}

HTShapeList DivFromConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void ReciprocalOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::reciprocal(inputs.at(0), stream_index(), outputs.at(0));
}

TensorList ReciprocalOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  // 1 / (x^2) = (1 / x) * (1 / x)
  auto ret = MulElewiseOp(_outputs.at(0), _outputs.at(0), g_op_meta)->output(0);
  ret = NegateOp(ret, g_op_meta)->output(0);
  ret = MulElewiseOp(ret, grad_outputs.at(0), g_op_meta.set_name(grad_name()))
          ->output(0);
  return {ret};
}

HTShapeList ReciprocalOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu
