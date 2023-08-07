#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

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
    grad_a = op->requires_grad(0) ? MakeMatMulGradientOp(grad_c, b, a, 0, false, true, std::move(grad_a_op_meta))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeMatMulGradientOp(a, grad_c, b, 1, true, false, std::move(grad_b_op_meta))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = MatMul(a^T, b)
    // grad_a = MatMul(b, grad_c^T), grad_b = MatMul(a, grad_c)
    grad_a = op->requires_grad(0) ? MakeMatMulGradientOp(b, grad_c, a, 1, false, true, std::move(grad_a_op_meta))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeMatMulGradientOp(a, grad_c, b, 1, false, false, std::move(grad_b_op_meta))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = MatMul(a, b^T)
    // grad_a = MatMul(grad_c, b), grad_b = MatMul(grad_c^T, a)
    grad_a = op->requires_grad(0) ? MakeMatMulGradientOp(grad_c, b, a, 0, false, false, std::move(grad_a_op_meta))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeMatMulGradientOp(grad_c, a, b, 0, true, false, std::move(grad_b_op_meta))
                                 : Tensor();
  } else {
    // case 4: c = MatMul(a^T, b^T)
    // grad_a = MatMul(b^T, grad_c^T), grad_b = MatMul(grad_c^T, a^T)
    grad_a = op->requires_grad(0) ? MakeMatMulGradientOp(b, grad_c, a, 1, true, true, std::move(grad_a_op_meta))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeMatMulGradientOp(grad_c, a, b, 0, true, true, std::move(grad_b_op_meta))
                                 : Tensor();
  }
  return {grad_a, grad_b};
}

void MatMulGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                                     RuntimeContext& runtime_ctx) const {
  const auto a = inputs.at(0);
  const auto b = inputs.at(1);
  const auto dim_a = a->ndim();
  const auto dim_b = b->ndim();
  if (dim_a == 0 || dim_b == 0) {
    outputs.at(0) = a * b;
  } else if (dim_b == 1 && trans_b()) {
    auto a_shape = a->shape();
    auto a_ = a;
    if (dim_a >= 2 && trans_a()) {
      std::iter_swap(a_shape.end() - 2, a_shape.end() - 1);
      a_ = NDArray::empty(a_shape, a->device(), a->dtype());
      auto ndims_a_ = HTAxes(dim_a);
      std::iota(ndims_a_.begin(), ndims_a_.end(), 0);
      std::iter_swap(ndims_a_.end() - 2, ndims_a_.end() - 1);
      a_ = NDArray::permute(a, ndims_a_, op->instantiation_ctx().stream_index, a_);
    }
    NDArray::matmul(NDArray::unsqueeze(a_, dim_a), NDArray::unsqueeze(b, 0), false, false,
                    op->instantiation_ctx().stream_index, outputs.front());
  } else {
    NDArray unreduced;
    unreduced = NDArray::matmul(a, b, trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, unreduced);
    const auto grad = inputs.at(grad_idx());
    const auto dst = inputs.at(2);
    const auto dim_grad = grad->ndim();
    const auto dim_dst = dst->ndim();
    if (dim_grad > dim_dst) {
      auto reduce_dims = HTAxes(dim_grad - dim_dst);
      std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                      hetu::impl::ReduceSum, unreduced, outputs.front(),
                                      reduce_dims.data(), reduce_dims.size(),
                                      op->instantiation_ctx().stream());
    } else {
      NDArray::copy(unreduced, op->instantiation_ctx().stream_index, outputs.front());
    }
  }
}

Tensor MakeMatMulOp(Tensor a, Tensor b, bool trans_a, bool trans_b,
                    OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  DataType input_type = DataType::FLOAT16;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(std::make_shared<MatMulOpImpl>(trans_a, trans_b),
                       std::move(inputs), std::move(op_meta))
    ->output(0);
}

Tensor MakeMatMulGradientOp(Tensor a, Tensor b, Tensor dst, int grad_idx,
                            bool trans_a, bool trans_b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b), std::move(dst)};
  return Graph::MakeOp(std::make_shared<MatMulGradientOpImpl>(trans_a, trans_b, grad_idx),
                      std::move(inputs), std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
