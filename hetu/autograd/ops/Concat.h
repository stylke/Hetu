#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class ConcatOpDef;
class ConcatOp;
class ConcatGradientOpDef;
class ConcatGradientOp;

class ConcatOpDef : public OperatorDef {
 private:
  friend class ConcatOp;
  struct constrcutor_access_key {};

 public:
  ConcatOpDef(const constrcutor_access_key&, Tensor inputA, Tensor inputB,
              size_t axis, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatOp), {inputA, inputB}, op_meta), _axis(axis) {
    HTShape shape;
    if (inputA->has_shape() && inputB->has_shape()) {
      shape = inputA->shape();
      if (shape[get_axis()] > 0) {
        shape[get_axis()] += inputB->shape(get_axis());
      }
    }
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    DeduceStates();
  }

  void DeduceStates() override;

  size_t get_axis() const {
    return _axis;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;
};

class ConcatOp final : public OpWrapper<ConcatOpDef> {
 public:
  ConcatOp(Tensor inputA, Tensor inputB, size_t axis,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatOpDef>(make_ptr<ConcatOpDef>(
      ConcatOpDef::constrcutor_access_key(), inputA, inputB, axis, op_meta)) {}
};

class ConcatGradientOpDef : public OperatorDef {
 private:
  friend class ConcatGradientOp;
  struct constrcutor_access_key {};

 public:
  ConcatGradientOpDef(const constrcutor_access_key&, Tensor input,
                      Tensor grad_output, size_t axis, size_t id,
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatGradientOp), {input, grad_output}, op_meta),
    _axis(axis),
    _id(id) {
    AddOutput(input->meta());
    DeduceStates();
  }

  void DeduceStates() override;

  size_t get_axis() const {
    return _axis;
  }

  size_t get_id() const {
    return _id;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;

  size_t _id;
};

class ConcatGradientOp final : public OpWrapper<ConcatGradientOpDef> {
 public:
  ConcatGradientOp(Tensor input, Tensor grad_output, size_t axis, size_t id,
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatGradientOpDef>(make_ptr<ConcatGradientOpDef>(
      ConcatGradientOpDef::constrcutor_access_key(), input, grad_output, axis,
      id, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
