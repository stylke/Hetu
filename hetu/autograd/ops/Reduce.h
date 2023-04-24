#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ReduceOpDef;
class ReduceOp;
class ReduceGradientOpDef;
class ReduceGradientOp;

class ReduceOpDef : public OperatorDef {
 private:
  friend class ReduceOp;
  struct constrcutor_access_key {};

 public:
  ReduceOpDef(const constrcutor_access_key&, Tensor input,
              const std::string& mode, const HTAxes& axes,
              const HTKeepDims& keepdims = {false},
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceOp), {input}, op_meta),
    _mode(mode),
    _axes(axes),
    _keepdims(keepdims) {
    HT_ASSERT(keepdims.size() == axes.size() || keepdims.size() == 1);
    if (keepdims.size() == 1) {
      int len = axes.size();
      bool keepdim = keepdims[0];
      for (int i = 1; i < len; ++i) {
        _keepdims.emplace_back(keepdim);
      }
    }
    HTShape output_shape;
    if (input->has_shape()) {
      int ndim = input->ndim();
      HTShape tmp_axes = axes;
      HTShape input_shape = input->shape();
      int len = axes.size();
      for (int i = 0; i < len; ++i) {
        if (tmp_axes[i] < 0) {
          tmp_axes[i] += ndim;
        }
        HT_ASSERT(tmp_axes[i] >= 0 && tmp_axes[i] < ndim)
          << "axes:" << tmp_axes[i] << " ,ndims:" << ndim;
        if (keepdims[i] == true)
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
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
    DeduceStates();
  }

  void DeduceStates() override;

  const HTShape& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  const std::string& mode() const {
    return _mode;
  }

  void set_axes(const HTShape& axes) {
    _axes = axes;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

  const HTShape& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTShape& get_grad_shape() const {
    return _grad_shape;
  }

  double get_grad_const() const {
    return _grad_const;
  }

  void set_grad_axes(const HTShape& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_shape(const HTShape& shape) {
    _grad_shape = shape;
  }

  void set_grad_const(double constant) {
    _grad_const = constant;
  }

  Operator grad, grad_;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTShape _axes;

  HTKeepDims _keepdims;

  HTShape _grad_add_axes;

  HTShape _grad_shape;

  double _grad_const;
};

class ReduceOp final : public OpWrapper<ReduceOpDef> {
 public:
  ReduceOp() : OpWrapper<ReduceOpDef>() {}
  ReduceOp(Tensor input, const std::string& mode, const HTAxes& axes,
           const HTKeepDims& keepdims = {false},
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceOpDef>(
      make_ptr<ReduceOpDef>(ReduceOpDef::constrcutor_access_key(), input, mode,
                            axes, keepdims, op_meta)) {}
};

class ReduceGradientOpDef : public OperatorDef {
 private:
  friend class ReduceGradientOp;
  struct constrcutor_access_key {};

 public:
  ReduceGradientOpDef(const constrcutor_access_key&, Tensor input,
                      Tensor ori_input, const HTShape& shape,
                      const std::string& mode,
                      const HTAxes& add_axes = HTAxes(),
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceGradientOp), {input, ori_input}, op_meta),
    grad_input(),
    _mode(mode),
    _shape(shape),
    _add_axes(add_axes) {
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    DeduceStates();
  }

  void DeduceStates() override;

  Operator grad_input;

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTShape& get_add_axes() const {
    return _add_axes;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_add_axes(const HTShape& shape) {
    _add_axes = shape;
  }

  const HTShape& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTKeepDims& get_grad_keep_dims() const {
    return _grad_keep_dims;
  }

  double get_const_value() const {
    return _constant;
  }

  std::string mode() const {
    return _mode;
  }

  void set_grad_axes(const HTShape& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_keep_dims(const HTKeepDims& keep_dims) {
    _grad_keep_dims = keep_dims;
  }

  void set_const_value(double constant) {
    _constant = constant;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTShape _shape;

  HTShape _add_axes;

  double _constant;

  HTShape _grad_add_axes;

  HTKeepDims _grad_keep_dims;
};

class ReduceGradientOp final : public OpWrapper<ReduceGradientOpDef> {
 public:
  ReduceGradientOp(Tensor input, Tensor ori_input, const HTShape& shape,
                   std::string mode, const HTAxes& add_axes = HTAxes(),
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceGradientOpDef>(make_ptr<ReduceGradientOpDef>(
      ReduceGradientOpDef::constrcutor_access_key(), input, ori_input, shape,
      mode, add_axes, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
