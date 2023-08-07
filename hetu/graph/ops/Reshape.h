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
    int64_t input_size = 1;
    HTShape input_shape = inputs[0]->shape();
    int64_t input_len = input_shape.size();
    int64_t idx = -1;
    size_t cnt = 0;
    int64_t output_size = 1;
    HTShape output_shape = get_output_shape();
    int64_t output_len = output_shape.size();
    for (size_t i = 0; i < input_len; ++i) {
      if (input_shape[i] == -1) {
        cnt = cnt + 1;
        HT_ASSERT(cnt != 2) << "Input shape has more than one '-1' dims. ";
      }
      input_size *= input_shape[i];
    }
    cnt = 0;
    for (int64_t i = 0; i < output_len; ++i) {
      if (output_shape[i] == -1) {
        idx = i;
        cnt = cnt + 1;
        HT_ASSERT(cnt != 2) << "Output shape has more than one '-1' dims. ";
      }
      output_size *= output_shape[i];
    }
    if (idx == -1) {
      HT_ASSERT(input_size == output_size) << "Invalid output size.";
    } else {
      output_size = output_size * (-1);
      HT_ASSERT(input_size % output_size == 0) << "Invalid output size." << input_shape << "," << output_shape
                                              << input_size << "," << output_size;
      output_shape[idx] = input_size / output_size;
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

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
                          OpMeta op_meta = OpMeta());

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

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }

};

Tensor MakeArrayReshapeGradientOp(Tensor grad_output, Tensor ori_input,
                                  OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
