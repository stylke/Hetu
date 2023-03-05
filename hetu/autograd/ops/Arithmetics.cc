#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace autograd {

std::pair<HTAxes, HTKeepDims> GradInfer(const HTShapeList& input_shapes) {
    HTShape output_shape = input_shapes[3];
  HTShape input_shape = input_shapes[2];
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim);
  size_t diff = ndim - input_shape.size();
  HTAxes add_axes(diff);
  HTKeepDims keep_dims(diff);
  size_t len = diff + input_shape.size();
  HTShape n_input_shape(len);
  for (size_t i = 0; i < diff; ++i) {
    add_axes[i] = i;
    keep_dims[i] = false;
    n_input_shape[i] = 1;
  }
  for (size_t i = diff; i < len; ++i) {
    n_input_shape[i] = input_shape[i - diff];
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (output_shape[i] == -1) {
      output_shape[i] = n_input_shape[i];
    }
    HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
    if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
      add_axes.emplace_back(i);
      keep_dims.emplace_back(true);
    }
  }
  return std::make_pair(add_axes, keep_dims);
}

void AddElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::add(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList AddElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_a = AddElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = AddElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};
}

void AddElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
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

void AddByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList AddByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void SubElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::sub(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList SubElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_a = SubElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = SubElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};  
}

void SubElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
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

void SubByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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

void SubFromConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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

void NegateOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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
//   auto grad_a = MulElewiseOp(grad_outputs.at(0), _inputs[1],
//                              g_op_meta.set_name(grad_name(0)))
//                   ->output(0);
//   auto grad_b = MulElewiseOp(grad_outputs.at(0), _inputs[0],
//                              g_op_meta.set_name(grad_name(1)))
//                   ->output(0);
  auto grad_a = MulElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = MulElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
//     HT_LOG_INFO << _inputs[0]->shape() << "\n" << _inputs[1]->shape()
//    << "\n" << grad_a->shape() << "\n" << grad_b->shape();
  return {grad_a, grad_b};
}

void MulElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
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

void MulByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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
//   //1 / b
//   auto dividend_grad = ReciprocalOp(_inputs.at(1), g_op_meta)->output(0);
//   auto grad_a = MulElewiseOp(dividend_grad, grad_outputs.at(0),
//                              g_op_meta.set_name(grad_name(0)))
//                   ->output(0);
//   // - a / (b^2) = - (a / b) / b
//   auto divisor_grad =
//     NegateOp(DivElewiseOp(_outputs[0], _inputs.at(1), g_op_meta)->output(0),
//              g_op_meta)
//       ->output(0);
//   auto grad_b = MulElewiseOp(divisor_grad, grad_outputs.at(0),
//                              g_op_meta.set_name(grad_name(1)))
//                   ->output(0);
  auto grad_a = DivElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = DivElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};
}

void DivElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
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

void DivByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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

void DivFromConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
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

void ReciprocalOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ReciprocalOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void AddElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::copy(inputs.at(0), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void AddElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList AddElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void SubElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0)
    NDArray::copy(inputs.at(0), stream_index(), unreduced);
  else 
    NDArray::neg(inputs.at(0), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void SubElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList SubElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void MulElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::mul(inputs.at(0), inputs.at(1), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void MulElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList MulElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void DivElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0) {
    NDArray::div(inputs.at(0), inputs.at(1), stream_index(), unreduced);
  }
  else {
    NDArray::mul(inputs.at(0), inputs.at(3), stream_index(), unreduced);
    NDArray::div(unreduced, inputs.at(2), stream_index(), unreduced);
    NDArray::neg(unreduced, stream_index(), unreduced); 
  }
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void DivElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList DivElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

} // namespace autograd
} // namespace hetu
