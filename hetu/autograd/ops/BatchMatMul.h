#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class BatchMatMulOpDef;
class BatchMatMulOp;

class BatchMatMulOpDef : public OperatorDef {
 private:
  friend class BatchMatMulOp;
  struct constrcutor_access_key {};

 public:
  BatchMatMulOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
                   bool trans_a = false, bool trans_b = false,
                   const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchMatMulOp), {a, b}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b) {
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() >= 2 && b->ndim() >= 2)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be more than 2. "
        << "Got " << a->ndim() << ", " << b->ndim() << ".";
      int64_t ndims = a->ndim();
      int64_t dim_a = a->shape(trans_a ? ndims - 2 : ndims - 1);
      int64_t dim_b = b->shape(trans_b ? ndims - 1 : ndims - 2);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be compatible. "
        << "Got " << dim_a << " vs. " << dim_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
    }
    HTShape shape = {};
    if (a->has_shape() && b->has_shape()) {
      int ndims = a->ndim();
      for (int i = 0; i < ndims - 2; ++i) {
        HT_ASSERT(a->shape(i) == b->shape(i));
        shape.emplace_back(a->shape(i));
      }
      shape.emplace_back(a->shape(trans_a ? ndims - 1 : ndims - 2));
      shape.emplace_back(b->shape(trans_b ? ndims - 2 : ndims - 1));
    }
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    DeduceStates();
  }

  void DeduceStates() override;

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;
};

class BatchMatMulOp final : public OpWrapper<BatchMatMulOpDef> {
 public:
  BatchMatMulOp(Tensor a, Tensor b, bool trans_a = false, bool trans_b = false,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchMatMulOpDef>(
      make_ptr<BatchMatMulOpDef>(BatchMatMulOpDef::constrcutor_access_key(), a,
                                 b, trans_a, trans_b, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
