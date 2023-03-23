#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ReduceOpImpl;
class ReduceOp;
class ReduceGradientOpImpl;
class ReduceGradientOp;

class ReduceOpImpl : public OpInterface {
protected:
  ReduceOpImpl(OpType&& op_type, ReductionType reduction = kMEAN, 
               const HTAxes& axes = {},
               const HTKeepDims& keepdims = {false},
               const OpMeta& op_meta = OpMeta())
  : OpInterface(std::move(op_type)),
    _axes(axes),
    _keepdims(keepdims),
    _reduction(reduction) {
  }

 public:
  ReduceOpImpl(ReductionType reduction, const HTAxes& axes = {},
               const HTKeepDims& keepdims = {false},
               const OpMeta& op_meta = OpMeta())
  : ReduceOpImpl(quote(ReduceOp), reduction, axes, keepdims, op_meta) {}

  const HTAxes& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  ReductionType reduction() const {
    return _reduction;
  }

  void set_axes(const HTAxes& axes) {
    _axes = axes;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape output_shape;
    if (inputs[0]->has_shape()) {
      int ndim = inputs[0]->ndim();
      HTShape tmp_axes = _axes;
      HTShape input_shape = inputs[0]->shape();
      int len = tmp_axes.size();
      for (int i = 0; i < len; ++i) {
        if (tmp_axes[i] < 0) {
          tmp_axes[i] += ndim;
        }
        HT_ASSERT(tmp_axes[i] >= 0 && tmp_axes[i] < ndim)
          << "axes:" << tmp_axes[i] << " ,ndims:" << ndim;
        if (_keepdims[i] == true)
          input_shape[tmp_axes[i]] = 1;
        else
          input_shape[tmp_axes[i]] = 0;
      }
      for (int i = 0; i < ndim; ++i) {
        if (input_shape[i] > 0)
          output_shape.emplace_back(input_shape[i]);
      }
      if (output_shape.size() == 0)
        output_shape.emplace_back(1);
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTAxes _axes;

  HTKeepDims _keepdims;

  ReductionType _reduction;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ReduceOpImpl&>(rhs);
      return (get_axes() == rhs_.get_axes()
              && get_keepdims() == rhs_.get_keepdims()); 
    }
    return false;
  }
};

Tensor MakeReduceOp(Tensor input, ReductionType reduction, const HTAxes& axes = {},
                    const HTKeepDims& keepdims = {false},
                    const OpMeta& op_meta = OpMeta());

Tensor MakeReduceOp(Tensor input, const std::string& mode, const HTAxes& axes = {},
                    const HTKeepDims& keepdims = {false},
                    const OpMeta& op_meta = OpMeta());

class ReduceGradientOpImpl : public OpInterface {
 public:
  ReduceGradientOpImpl(const HTShape& shape,
                       ReductionType reduction = kMEAN,
                       const HTAxes add_axes = HTAxes(),
                       const HTKeepDims& keepdims = HTKeepDims(),
                       const double value = 0,
                       const OpMeta& op_meta = OpMeta())
  : OpInterface(quote(ReduceGradientOp)),
    _shape(shape),
    _axes(add_axes),
    _keepdims(keepdims),
    _reduction(reduction),
    _constant(value) {
  }

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTAxes& get_axes() const {
    return _axes;
  }

  const HTAxes& get_add_axes() const {
    return _add_axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_axes(const HTAxes& shape) {
    _axes = shape;
  }

  void set_add_axes(const HTAxes& shape) {
    _add_axes = shape;
  }

  double get_const_value() const {
    return _constant;
  }

  ReductionType reduction() const {
    return _reduction;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

  void set_const_value(double constant) {
    _constant = constant;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _shape;

  HTAxes _add_axes;

  HTAxes _axes;

  HTKeepDims _keepdims;

  double _constant;

  ReductionType _reduction;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ReduceGradientOpImpl&>(rhs);
      return (get_axes() == rhs_.get_axes()
              && get_keepdims() == rhs_.get_keepdims()
              && get_add_axes() == rhs_.get_add_axes()
              && get_const_value() == rhs_.get_const_value()); 
    }
    return false;
  }
};

Tensor MakeReduceGradientOp(Tensor input, Tensor ori_output, Tensor ori_input, const HTShape& shape,
                            ReductionType reduction, const HTAxes add_axes = HTAxes(), const HTKeepDims& keepdims = HTKeepDims(),
                            const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
