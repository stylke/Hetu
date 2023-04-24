#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class LayerNormOpDef;
class LayerNormOp;
class LayerNormGradientOpDef;
class LayerNormGradientOp;
class LayerNormGradientofDataOpDef;
class LayerNormGradientofDataOp;
class LayerNormGradientofScaleOpDef;
class LayerNormGradientofScaleOp;
class LayerNormGradientofBiasOpDef;
class LayerNormGradientofBiasOp;

class LayerNormOpDef : public OperatorDef {
 private:
  friend class LayerNormOp;
  struct constrcutor_access_key {};

 public:
  LayerNormOpDef(const constrcutor_access_key&, Tensor input, Tensor ln_scale,
                 Tensor ln_bias, double eps = 0.01,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormOp), {input, ln_scale, ln_bias}, op_meta),
    _eps(eps) {
    AddOutput(input->meta());
    DeduceStates();
  }

  void DeduceStates() override;

  double get_momentum() const {
    return _momentum;
  }

  double get_eps() const {
    return _eps;
  }

  HTShape get_shape() const {
    return _shape;
  }

  void set_shape(HTShape shape) {
    _shape = shape;
  }

  NDArray save_mean;

  NDArray save_var;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _momentum;

  double _eps;

  HTShape _shape;
};

class LayerNormOp final : public OpWrapper<LayerNormOpDef> {
 public:
  LayerNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias, double eps = 0.01,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormOpDef>(
      make_ptr<LayerNormOpDef>(LayerNormOpDef::constrcutor_access_key(), input,
                               bn_scale, bn_bias, eps, op_meta)) {}
};

class LayerNormGradientOpDef : public OperatorDef {
 private:
  friend class LayerNormGradientOp;
  struct constrcutor_access_key {};

 public:
  LayerNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                         Tensor input, Tensor bn_scale,
                         LayerNormOp forward_node, double eps,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormGradientOp), {output_grad, input, bn_scale},
                op_meta),
    _forward_node(forward_node),
    _eps(eps) {
    AddOutput(output_grad->meta());
  }

  LayerNormOp get_forward_node() const {
    return _forward_node;
  }

  double get_eps() const {
    return _eps;
  }

  NDArray tmp_gradient_bn_arr;

  NDArray tmp_gradient_bn_scale;

  NDArray tmp_gradient_bn_bias;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  LayerNormOp _forward_node;

  double _eps;
};

class LayerNormGradientOp final : public OpWrapper<LayerNormGradientOpDef> {
 public:
  LayerNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                      LayerNormOp forward_node, double eps,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormGradientOpDef>(make_ptr<LayerNormGradientOpDef>(
      LayerNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      bn_scale, forward_node, eps, op_meta)) {}
};

/*————————————LayerNormGradientofDataOp————————————————————*/
class LayerNormGradientofDataOpDef : public OperatorDef {
 private:
  friend class LayerNormGradientofDataOp;
  struct constrcutor_access_key {};

 public:
  LayerNormGradientofDataOpDef(const constrcutor_access_key&,
                               Tensor bn_gradient, Tensor input,
                               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormGradientofDataOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class LayerNormGradientofDataOp final
: public OpWrapper<LayerNormGradientofDataOpDef> {
 public:
  LayerNormGradientofDataOp(Tensor bn_gradient, Tensor input,
                            const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormGradientofDataOpDef>(
      make_ptr<LayerNormGradientofDataOpDef>(
        LayerNormGradientofDataOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

/*————————————LayerNormGradientofScaleOp————————————————————*/
class LayerNormGradientofScaleOpDef : public OperatorDef {
 private:
  friend class LayerNormGradientofScaleOp;
  struct constrcutor_access_key {};

 public:
  LayerNormGradientofScaleOpDef(const constrcutor_access_key&,
                                Tensor bn_gradient, Tensor input,
                                const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormGradientofScaleOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class LayerNormGradientofScaleOp final
: public OpWrapper<LayerNormGradientofScaleOpDef> {
 public:
  LayerNormGradientofScaleOp(Tensor bn_gradient, Tensor input,
                             const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormGradientofScaleOpDef>(
      make_ptr<LayerNormGradientofScaleOpDef>(
        LayerNormGradientofScaleOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

/*————————————LayerNormGradientofBiasOp————————————————————*/
class LayerNormGradientofBiasOpDef : public OperatorDef {
 private:
  friend class LayerNormGradientofBiasOp;
  struct constrcutor_access_key {};

 public:
  LayerNormGradientofBiasOpDef(const constrcutor_access_key&,
                               Tensor bn_gradient, Tensor input,
                               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LayerNormGradientofBiasOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class LayerNormGradientofBiasOp final
: public OpWrapper<LayerNormGradientofBiasOpDef> {
 public:
  LayerNormGradientofBiasOp(Tensor bn_gradient, Tensor input,
                            const OpMeta& op_meta = OpMeta())
  : OpWrapper<LayerNormGradientofBiasOpDef>(
      make_ptr<LayerNormGradientofBiasOpDef>(
        LayerNormGradientofBiasOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
