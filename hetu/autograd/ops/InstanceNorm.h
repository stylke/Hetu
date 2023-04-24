#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class InstanceNormOpDef;
class InstanceNormOp;
class InstanceNormGradientOpDef;
class InstanceNormGradientOp;

class InstanceNormOpDef : public OperatorDef {
 private:
  friend class InstanceNormOp;
  struct constrcutor_access_key {};

 public:
  InstanceNormOpDef(const constrcutor_access_key&, Tensor input,
                    double eps = 1e-7, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InstanceNormOp), {input}, op_meta), _eps(eps) {
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

class InstanceNormOp final : public OpWrapper<InstanceNormOpDef> {
 public:
  InstanceNormOp(Tensor input, double eps = 1e-7,
                 const OpMeta& op_meta = OpMeta())
  : OpWrapper<InstanceNormOpDef>(make_ptr<InstanceNormOpDef>(
      InstanceNormOpDef::constrcutor_access_key(), input, eps, op_meta)) {}
};

class InstanceNormGradientOpDef : public OperatorDef {
 private:
  friend class InstanceNormGradientOp;
  struct constrcutor_access_key {};

 public:
  InstanceNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                            Tensor input, InstanceNormOp forward_node,
                            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InstanceNormGradientOp), {output_grad, input}, op_meta),
    _forward_node(forward_node) {
    AddOutput(output_grad->meta());
    DeduceStates();
  }

  void DeduceStates() override;

  InstanceNormOp get_forward_node() const {
    return _forward_node;
  }

  double get_eps() const {
    return _eps;
  }

  NDArray tmp_gradient_bn_scale;

  NDArray tmp_gradient_bn_bias;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  InstanceNormOp _forward_node;

  double _eps;
};

class InstanceNormGradientOp final
: public OpWrapper<InstanceNormGradientOpDef> {
 public:
  InstanceNormGradientOp(Tensor output_grad, Tensor input,
                         InstanceNormOp forward_node,
                         const OpMeta& op_meta = OpMeta())
  : OpWrapper<InstanceNormGradientOpDef>(make_ptr<InstanceNormGradientOpDef>(
      InstanceNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      forward_node, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
