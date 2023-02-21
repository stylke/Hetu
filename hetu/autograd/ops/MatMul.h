#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

#include "hetu/impl/communication/comm_group.h"
using namespace hetu::impl::comm;

namespace hetu {
namespace autograd {

class MatMulOpDef;
class MatMulOp;

class MatMulOpDef : public OperatorDef {
 private:
  friend class MatMulOp;
  struct constrcutor_access_key {};

 public:
  MatMulOpDef(const constrcutor_access_key&, Tensor a, Tensor b,
              bool trans_a = false, bool trans_b = false,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MatMulOp), {a, b}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b) {
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be 2. "
        << "Got " << a->ndim() << ", " << b->ndim() << ".";
      int64_t dim_a = a->shape(trans_a ? 0 : 1);
      int64_t dim_b = b->shape(trans_b ? 1 : 0);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be compatible. "
        << "Got " << dim_a << " vs. " << dim_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
    }
    HTShape shape = {-1, -1};
    if (a->has_shape())
      shape[0] = a->shape(trans_a ? 1 : 0);
    if (b->has_shape())
      shape[1] = b->shape(trans_b ? 0 : 1);
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));

    // 在node定义的时候就做ds的推导
    ForwardDeduceStates(); // inputs.ds -> outputs.ds
    // test: 这里假设每个op的output只有一个, 简单测试输出的ds, 如果op的output不只一个, 则只输出第0个的情况
    auto ds = _outputs[0]->get_distributed_states();
    auto local_device = GetLocalDevice();
    HT_LOG_DEBUG << local_device << ": " << name() << " definition, output[0] states: " << ds.print_states() << ", shape: " << _outputs[0]->shape();
  }

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

  void ForwardDeduceStates();
  DistributedStates BackwardDeduceStates(int32_t index);

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;
};

class MatMulOp final : public OpWrapper<MatMulOpDef> {
 public:
  MatMulOp(Tensor a, Tensor b, bool trans_a = false, bool trans_b = false,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<MatMulOpDef>(make_ptr<MatMulOpDef>(
      MatMulOpDef::constrcutor_access_key(), a, b, trans_a, trans_b, op_meta)) {
  }
};

} // namespace autograd
} // namespace hetu
