#include "hetu/graph/ops/Linear.h"
#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LinearOpImpl::DoCompute(Operator& op,const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Linear,
  //                              inputs.at(0), trans_a(), inputs.at(1), trans_b(),
  //                              inputs.at(2), outputs.at(0), op->instantiation_ctx().stream());
  NDArray::linear(inputs.at(0), inputs.at(1), inputs.at(2), trans_a(), trans_b(),
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList LinearOpImpl::DoGradient(Operator& op,const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = op->grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = Linear(a, b)
    // grad_a = Linear(grad_c, b^T), grad_b = Linear(a^T, grad_c)
    grad_a = op->require_grad(0) ? MakeMatMulOp(grad_c, b, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeMatMulOp(a, grad_c, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = Linear(a^T, b)
    // grad_a = Linear(b, grad_c^T), grad_b = Linear(a, grad_c)
    grad_a = op->require_grad(0) ? MakeMatMulOp(b, grad_c, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeMatMulOp(a, grad_c, false, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = Linear(a, b^T)
    // grad_a = Linear(grad_c, b), grad_b = Linear(grad_c^T, a)
    grad_a = op->require_grad(0) ? MakeMatMulOp(grad_c, b, false, false, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeMatMulOp(grad_c, a, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else {
    // case 4: c = Linear(a^T, b^T)
    // grad_a = Linear(b^T, grad_c^T), grad_b = Linear(grad_c^T, a^T)
    grad_a = op->require_grad(0) ? MakeMatMulOp(b, grad_c, true, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->require_grad(1) ? MakeMatMulOp(grad_c, a, true, true, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  }
  Tensor grad_bias = op->require_grad(2) ? MakeReduceOp(grad_outputs.at(0), ReductionType::SUM, {0}, {false},
                                           g_op_meta.set_name(op->grad_name(2)))
                                         : Tensor();
  return {grad_a, grad_b, grad_bias};
}

HTShapeList LinearOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  HT_ASSERT(a.size() == 2 && b.size() == 2 &&
            a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
    << "Invalid input shapes for " << type() << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
}

Tensor MakeLinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a,
                    bool trans_b, const OpMeta& op_meta) {
  return Graph::MakeOp(
        std::make_shared<LinearOpImpl>(trans_a, trans_b),
        {std::move(a), std::move(b)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
