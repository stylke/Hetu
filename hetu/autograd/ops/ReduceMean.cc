#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/BroadcastShape.h"
#include "hetu/autograd/ops/ReduceMean.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ReduceMeanOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::ReduceMean, inputs.at(0),
    outputs.at(0), get_axes().data(), get_axes().size(), stream());
}

TensorList ReduceMeanOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad =
    BroadcastShapeOp(grad_outputs.at(0), HTShape(), HTAxes(), g_op_meta)
      ->output(0);
  grad = MulByConstOp(grad, 0, g_op_meta.set_name(grad_name()))->output(0);
  return {grad};
}

HTShapeList ReduceMeanOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape input_shape = input_shapes.at(0);
  int ndim = input_shape.size();
  int64_t mean_multiplier = 1;
  HTShape axes = get_axes();
  int len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  set_grad_shape(input_shape);
  // BroadcastShapeOp& grad_b = reinterpret_cast<BroadcastShapeOp&>(grad);
  // MulByConstOp grad_a = *(MulByConstOp*)&grad_;
  // if (grad_b)
  //   grad_b->set_shape(input_shape);
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
  set_grad_axes(add_axes);
  set_grad_const(1.0 / mean_multiplier);
  // if (grad_b)
  //   grad_b->set_add_axes(add_axes);
  // if (grad_a)
  //   grad_a->set_const_value(1.0/mean_multiplier);
  HTShape output_shape(0);
  for (int i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  return {output_shape};
}

} // namespace autograd
} // namespace hetu
