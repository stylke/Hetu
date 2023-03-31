#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ConcatOpImpl;
class ConcatOp;
class ConcatGradientOpImpl;
class ConcatGradientOp;

class ConcatOpImpl : public OpInterface {
 public:
  ConcatOpImpl(size_t axis, const OpMeta& op_meta = OpMeta())
  : OpInterface(quote(ConcatOp)), _axis(axis) {
  }

  size_t get_axis() const {
    return _axis;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape shape;
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      for (size_t i = 0; i < inputs[0]->ndim(); ++i) {
        if (i != get_axis())
          HT_ASSERT(inputs[0]->shape(i) == inputs[1]->shape(i))
          << "inputA and inputB has different size at dim " << i
          << ", inputA has " << inputs[0]->shape(i) <<  ",inputB has "
          << inputs[1]->shape(i);
        }
      HT_ASSERT(inputs[0]->shape(get_axis()) >= 0 && inputs[1]->shape(get_axis()) >= 0);
      shape = inputs[0]->shape();
      shape[get_axis()] += inputs[1]->shape(get_axis());
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis());
    }
    return false;
  }
};


Tensor MakeConcatOp(Tensor inputA, Tensor inputB, size_t axis,
                    const OpMeta& op_meta = OpMeta());

class ConcatGradientOpImpl : public OpInterface {
 public:
  ConcatGradientOpImpl(size_t axis, size_t id,
                       const OpMeta& op_meta = OpMeta())
  : OpInterface(quote(ConcatGradientOp)),
    _axis(axis),
    _id(id) {
  }

  size_t get_axis() const {
    return _axis;
  }

  size_t get_id() const {
    return _id;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }


  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;

  size_t _id;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatGradientOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis()
              && get_id() == rhs_.get_id());
    }
    return false;
  }
};

Tensor MakeConcatGradientOp(Tensor input, Tensor grad_output, size_t axis, size_t id,
                            const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
