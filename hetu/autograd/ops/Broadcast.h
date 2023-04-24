#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BroadcastOpDef;
class BroadcastOp;

class BroadcastOpDef : public OperatorDef {
 private:
  friend class BroadcastOp;
  struct constrcutor_access_key {};

 public:
  BroadcastOpDef(const constrcutor_access_key&, Tensor input, Tensor output,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastOp), {input, output}, op_meta), _mode(0) {
    AddOutput(output->meta());
    // DeduceStates();
  }

  BroadcastOpDef(const constrcutor_access_key&, Tensor input,
                 const HTShape& shape, const HTShape& add_axes = HTShape(),
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastOp), {input}, op_meta),
    _mode(1),
    _shape(shape),
    _add_axes(add_axes) {
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

  void DeduceStates() override;

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTAxes& get_add_axes() const {
    return _add_axes;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_add_axes(const HTAxes& shape) {
    _add_axes = shape;
  }

  const HTAxes& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTKeepDims& get_grad_keep_dims() const {
    return _grad_keep_dims;
  }

  int mode() const {
    return _mode;
  }

  void set_grad_axes(const HTAxes& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_keep_dims(const HTKeepDims& keep_dims) {
    _grad_keep_dims = keep_dims;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int _mode; // 0 = Broadcast, 1 = BroadcastShape

  HTShape _shape;

  HTShape _add_axes;

  HTShape _grad_add_axes;

  HTKeepDims _grad_keep_dims;
};

class BroadcastOp final : public OpWrapper<BroadcastOpDef> {
 public:
  BroadcastOp(Tensor input, Tensor output, const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastOpDef>(make_ptr<BroadcastOpDef>(
      BroadcastOpDef::constrcutor_access_key(), input, output, op_meta)) {}

  BroadcastOp(Tensor input, const HTShape& shape,
              const HTShape& add_axes = HTShape(),
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastOpDef>(
      make_ptr<BroadcastOpDef>(BroadcastOpDef::constrcutor_access_key(), input,
                               shape, add_axes, op_meta)) {}
};

class BroadcastGradientOpDef : public OperatorDef {
 private:
  friend class BroadcastGradientOp;
  struct constrcutor_access_key {};

 public:
  BroadcastGradientOpDef(const constrcutor_access_key&, Tensor input,
                         Tensor ori_input, const HTShape& axes,
                         const HTKeepDims& keepdims,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastGradientOp), {input, ori_input}, op_meta),
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
    HTShape output_shape(0);
    for (int i = 0; i < ndim; ++i) {
      if (input_shape[i] > 0)
        output_shape.emplace_back(input_shape[i]);
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(1);
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
  }

  const HTShape& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
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

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _axes;

  HTKeepDims _keepdims;

  HTShape _grad_add_axes;

  HTShape _grad_shape;

  double _grad_const;
};

class BroadcastGradientOp final : public OpWrapper<BroadcastGradientOpDef> {
 public:
  BroadcastGradientOp(Tensor input, Tensor ori_input, const HTShape& axes,
                      const HTKeepDims& keepdims,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastGradientOpDef>(make_ptr<BroadcastGradientOpDef>(
      BroadcastGradientOpDef::constrcutor_access_key(), input, ori_input, axes,
      keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
