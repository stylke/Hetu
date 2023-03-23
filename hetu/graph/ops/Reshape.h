#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ArrayReshapeOpImpl;
class ArrayReshapeOp;
class ArrayReshapeGradientOpImpl;
class ArrayReshapeGradientOp;

class ArrayReshapeOpImpl : public OpInterface {
 private:
  friend class ArrayReshapeOp;
  struct constrcutor_access_key {};

 public:
  ArrayReshapeOpImpl(const HTShape& output_shape)
  : OpInterface(quote(ArrayReshapeOp)),
    _output_shape(output_shape) {
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(get_output_shape())
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _output_shape;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ArrayReshapeOpImpl&>(rhs);
      return (get_output_shape() == rhs_.get_output_shape());
    }
    return false;
  }
};

Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          const OpMeta& op_meta = OpMeta());

class ArrayReshapeGradientOpImpl : public OpInterface {

 public:
  ArrayReshapeGradientOpImpl()
  : OpInterface(quote(ArrayReshapeGradientOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[1]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }

};

Tensor MakeArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input,
                                  const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
