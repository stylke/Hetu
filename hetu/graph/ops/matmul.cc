#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

TensorList MatMulOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto grad_a_op_meta = op->grad_op_meta().set_name(op->grad_name(0));
  auto grad_b_op_meta = op->grad_op_meta().set_name(op->grad_name(1));
  if (!trans_a() && !trans_b()) {
    // case 1: c = MatMul(a, b)
    // grad_a = MatMul(grad_c, b^T), grad_b = MatMul(a^T, grad_c)
    grad_a = MakeMatMulOp(grad_c, b, false, true, std::move(grad_a_op_meta));
    grad_b = MakeMatMulOp(a, grad_c, true, false, std::move(grad_b_op_meta));
  } else if (trans_a() && !trans_b()) {
    // case 2: c = MatMul(a^T, b)
    // grad_a = MatMul(b, grad_c^T), grad_b = MatMul(a, grad_c)
    grad_a = MakeMatMulOp(b, grad_c, false, true, std::move(grad_a_op_meta));
    grad_b = MakeMatMulOp(a, grad_c, false, false, std::move(grad_b_op_meta));
  } else if (!trans_a() && trans_b()) {
    // case 3: c = MatMul(a, b^T)
    // grad_a = MatMul(grad_c, b), grad_b = MatMul(grad_c^T, a)
    grad_a = MakeMatMulOp(grad_c, b, false, false, std::move(grad_a_op_meta));
    grad_b = MakeMatMulOp(grad_c, a, true, false, std::move(grad_b_op_meta));
  } else {
    // case 4: c = MatMul(a^T, b^T)
    // grad_a = MatMul(b^T, grad_c^T), grad_b = MatMul(grad_c^T, a^T)
    grad_a = MakeMatMulOp(b, grad_c, true, true, std::move(grad_a_op_meta));
    grad_b = MakeMatMulOp(grad_c, a, true, true, std::move(grad_b_op_meta));
  }
  return {grad_a, grad_b};
}

Tensor MakeMatMulOp(Tensor a, Tensor b, bool trans_a, bool trans_b,
                    OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<MatMulOpImpl>(trans_a, trans_b),
                       {std::move(a), std::move(b)}, std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
