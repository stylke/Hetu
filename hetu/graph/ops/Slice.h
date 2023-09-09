#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SliceOpImpl;
class SliceOp;
class SliceGradientOpImpl;
class SliceGradientOp;

class SliceOpImpl : public OpInterface {
 public:
  SliceOpImpl(const HTShape& begin_pos, const HTShape& output_shape)
  : OpInterface(quote(SliceOp)),
    _begin_pos(begin_pos),
    _output_shape(output_shape) {
  }
  
  uint64_t op_indicator() const noexcept override {
    return SLICE_OP;
  }

  HTShape get_begin_pos() const {
    return _begin_pos;
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT(inputs[0]->ndim() == _begin_pos.size() && 
              inputs[0]->ndim() == _output_shape.size());
    int len = _begin_pos.size();
    for (int i = 0; i < len; ++i) {
      HT_ASSERT(_begin_pos[i] >= 0 && (_output_shape[i] > 0 || _output_shape[i] == -1));
      HT_ASSERT(_begin_pos[i] + _output_shape[i] <= inputs[0]->shape(i));
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(_output_shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _begin_pos;

  HTShape _output_shape;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SliceOpImpl&>(rhs);
      return (get_begin_pos() == rhs_.get_begin_pos()
              && get_output_shape() == rhs_.get_output_shape());
    }
    return false;
  }
};

Tensor MakeSliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                   OpMeta op_meta = OpMeta());

class SliceGradientOpImpl : public OpInterface {

 public:
  SliceGradientOpImpl(const HTShape& begin_pos,
                      const HTShape& output_shape)
  : OpInterface(quote(SliceGradientOp)),
    _begin_pos(begin_pos),
    _output_shape(output_shape) {
  }

  HTShape get_begin_pos() const {
    return _begin_pos;
  }

  HTShape get_output_shape() const {
    return _output_shape;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[2]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _begin_pos;

  HTShape _output_shape;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SliceOpImpl&>(rhs);
      return (get_begin_pos() == rhs_.get_begin_pos()
              && get_output_shape() == rhs_.get_output_shape());
    }
    return false;
  }
};

Tensor MakeSliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                           const HTShape& begin_pos, const HTShape& output_shape,
                           OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
