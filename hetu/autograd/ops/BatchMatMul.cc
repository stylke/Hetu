#include "hetu/autograd/ops/BatchMatMul.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BatchMatMulOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::BatchMatMul, inputs.at(0),
    trans_a(), inputs.at(1), trans_b(), outputs.at(0), stream());
}

TensorList BatchMatMulOpDef::DoGradient(const TensorList& grad_outputs) {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = _inputs[0];
  Tensor& b = _inputs[1];
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = BatchMatMul(a, b)
    // grad_a = BatchMatMul(grad_c, b^T), grad_b = BatchMatMul(a^T, grad_c)
    grad_a =
      BatchMatMulOp(grad_c, b, false, true, g_op_meta.set_name(grad_name(0)))
        ->output(0);
    grad_b =
      BatchMatMulOp(a, grad_c, true, false, g_op_meta.set_name(grad_name(1)))
        ->output(0);
  } else if (trans_a() && !trans_b()) {
    // case 2: c = BatchMatMul(a^T, b)
    // grad_a = BatchMatMul(b, grad_c^T), grad_b = BatchMatMul(a, grad_c)
    grad_a =
      BatchMatMulOp(b, grad_c, false, true, g_op_meta.set_name(grad_name(0)))
        ->output(0);
    grad_b =
      BatchMatMulOp(a, grad_c, false, false, g_op_meta.set_name(grad_name(1)))
        ->output(0);
  } else if (!trans_a() && trans_b()) {
    // case 3: c = BatchMatMul(a, b^T)
    // grad_a = BatchMatMul(grad_c, b), grad_b = BatchMatMul(grad_c^T, a)
    grad_a =
      BatchMatMulOp(grad_c, b, false, false, g_op_meta.set_name(grad_name(0)))
        ->output(0);
    grad_b =
      BatchMatMulOp(grad_c, a, true, false, g_op_meta.set_name(grad_name(1)))
        ->output(0);
  } else {
    // case 4: c = BatchMatMul(a^T, b^T)
    // grad_a = BatchMatMul(b^T, grad_c^T), grad_b = BatchMatMul(grad_c^T, a^T)
    grad_a =
      BatchMatMulOp(b, grad_c, true, true, g_op_meta.set_name(grad_name(0)))
        ->output(0);
    grad_b =
      BatchMatMulOp(grad_c, a, true, true, g_op_meta.set_name(grad_name(1)))
        ->output(0);
  }
  return {grad_a, grad_b};
}

HTShapeList BatchMatMulOpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  int ndims = a.size() - 2;
  HT_ASSERT(a.size() >= 2 && b.size() >= 2 && a.size() == b.size() &&
            a.at(trans_a() ? ndims + 0 : ndims + 1) ==
              b.at(trans_b() ? ndims + 1 : ndims + 0))
    << "Invalid input shapes for " << type() << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  HTShape shape = {};
  for (int i = 0; i < ndims; ++i) {
    HT_ASSERT(a[i] == b[i]);
    shape.emplace_back(a[i]);
  }
  shape.emplace_back(a.at(trans_a() ? ndims + 1 : ndims));
  shape.emplace_back(b.at(trans_b() ? ndims : ndims + 1));
  return {shape};
}

} // namespace autograd
} // namespace hetu
