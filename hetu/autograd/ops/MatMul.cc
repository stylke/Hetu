#include "hetu/autograd/ops/MatMul.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void MatMulOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::MatMul,
                               inputs.at(0), trans_a(), inputs.at(1), trans_b(),
                               outputs.at(0), stream());
}

TensorList MatMulOpDef::DoGradient(const TensorList& grad_outputs) {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = _inputs[0];
  Tensor& b = _inputs[1];
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = MatMul(a, b)
    // grad_a = MatMul(grad_c, b^T), grad_b = MatMul(a^T, grad_c)
    grad_a = MatMulOp(grad_c, b, false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(a, grad_c, true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (trans_a() && !trans_b()) {
    // case 2: c = MatMul(a^T, b)
    // grad_a = MatMul(b, grad_c^T), grad_b = MatMul(a, grad_c)
    grad_a = MatMulOp(b, grad_c, false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(a, grad_c, false, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (!trans_a() && trans_b()) {
    // case 3: c = MatMul(a, b^T)
    // grad_a = MatMul(grad_c, b), grad_b = MatMul(grad_c^T, a)
    grad_a = MatMulOp(grad_c, b, false, false, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(grad_c, a, true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else {
    // case 4: c = MatMul(a^T, b^T)
    // grad_a = MatMul(b^T, grad_c^T), grad_b = MatMul(grad_c^T, a^T)
    grad_a = MatMulOp(b, grad_c, true, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMulOp(grad_c, a, true, true, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  }
  return {grad_a, grad_b};
}

void MatMulOpDef::DoInferMeta() {
  auto a = _inputs[0];
  auto b = _inputs[1];
  if (a->has_shape() && b->has_shape()) {
    HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "Dimensions must be 2. "
      << "Got " << a->ndim() << ", " << b->ndim() << ".";
    int64_t dim_a = a->shape(trans_a() ? 0 : 1);
    int64_t dim_b = b->shape(trans_b() ? 1 : 0);
    HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "Dimensions must be compatible. "
      << "Got " << dim_a << " vs. " << dim_b << ". "
      << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
  }
  HTShape shape = {-1, -1};
  if (a->has_shape())
    shape[0] = a->shape(trans_a() ? 1 : 0);
  if (b->has_shape())
    shape[1] = b->shape(trans_b() ? 0 : 1);
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList MatMulOpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  HT_ASSERT(a.size() == 2 && b.size() == 2 &&
            a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
    << "Failed to infer shape for the \"" << type() << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "Invalid input shapes: " << a << " (transpose_a = " << trans_a()
    << ") vs. " << b << " (transpose_b = " << trans_b() << "). ";
  return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
}

} // namespace autograd
} // namespace hetu
