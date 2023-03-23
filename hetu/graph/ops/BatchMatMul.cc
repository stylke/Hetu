#include "hetu/graph/ops/BatchMatMul.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void BatchMatMulOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& runtime_ctx) const {
  NDArray::bmm(inputs.at(0), inputs.at(1), trans_a(), trans_b(),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList BatchMatMulOpImpl::DoGradient(Operator& op,
                                         const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = op->grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = BatchMatMul(a, b)
    // grad_a = BatchMatMul(grad_c, b^T), grad_b = BatchMatMul(a^T, grad_c)
    grad_a = op->require_grad(0) ? MakeBatchMatMulOp(grad_c, b, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeBatchMatMulOp(a, grad_c, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = BatchMatMul(a^T, b)
    // grad_a = BatchMatMul(b, grad_c^T), grad_b = BatchMatMul(a, grad_c)
    grad_a = op->require_grad(0) ? MakeBatchMatMulOp(b, grad_c, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeBatchMatMulOp(a, grad_c, false, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = BatchMatMul(a, b^T)
    // grad_a = BatchMatMul(grad_c, b), grad_b = BatchMatMul(grad_c^T, a)
    grad_a = op->require_grad(0) ? MakeBatchMatMulOp(grad_c, b, false, false, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeBatchMatMulOp(grad_c, a, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else {
    // case 4: c = BatchMatMul(a^T, b^T)
    // grad_a = BatchMatMul(b^T, grad_c^T), grad_b = BatchMatMul(grad_c^T, a^T)
    grad_a = op->require_grad(0) ? MakeBatchMatMulOp(b, grad_c, true, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeBatchMatMulOp(grad_c, a, true, true, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  }
  return {grad_a, grad_b};
}


HTShapeList BatchMatMulOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& runtime_ctx) const {
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
    HT_ASSERT(a[i] == b[i])
    << op->name() << ",a:" << a << ",b:" << b;
    shape.emplace_back(a[i]);
  }
  shape.emplace_back(a.at(trans_a() ? ndims + 1 : ndims));
  shape.emplace_back(b.at(trans_b() ? ndims : ndims + 1));
  return {shape};
}

Tensor MakeBatchMatMulOp(Tensor a, Tensor b, bool trans_a, bool trans_b,
                         const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<BatchMatMulOpImpl>(trans_a, trans_b),
          {std::move(a), std::move(b)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
