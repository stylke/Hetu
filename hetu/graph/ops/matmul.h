#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class MatMulOpImpl final : public OpInterface {
 public:
  MatMulOpImpl(bool trans_a, bool trans_b)
  : OpInterface(quote(MatMulOp)), _trans_a(trans_a), _trans_b(trans_b) {}

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    const Tensor& a = inputs.at(0);
    const Tensor& b = inputs.at(1);
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be 2. "
        << "Got " << a->ndim() << ", " << b->ndim() << ".";
      int64_t dim_a = a->shape(trans_a() ? 0 : 1);
      int64_t dim_b = b->shape(trans_b() ? 1 : 0);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << dim_a << " vs. " << dim_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
    }
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape shape = {-1, -1};
    if (a->has_shape())
      shape[0] = a->shape(trans_a() ? 1 : 0);
    if (b->has_shape())
      shape[1] = b->shape(trans_b() ? 0 : 1);
    return {NDArrayMeta().set_dtype(a->dtype()).set_shape(shape)};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    const HTShape& a = input_shapes.at(0);
    const HTShape& b = input_shapes.at(1);
    HT_ASSERT(a.size() == 2 && b.size() == 2 &&
              a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
      << "Failed to infer shape for \"MatMul\" op: "
      << "Invalid input shapes: " << a << " (transpose_a = " << trans_a()
      << ") vs. " << b << " (transpose_b = " << trans_b() << "). ";
    return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    NDArray::matmul(inputs.at(0), inputs.at(1), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.front());
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MatMulOpImpl&>(rhs);
      return trans_a() == rhs_.trans_a() && trans_b() == rhs_.trans_b();
    }
    return false;
  }

  bool trans_a() const {
    return _trans_a;
  }

  bool trans_b() const {
    return _trans_b;
  }

 protected:
  bool _trans_a;
  bool _trans_b;
};

Tensor MakeMatMulOp(Tensor a, Tensor b, bool trans_a = false,
                    bool trans_b = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
