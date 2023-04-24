#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BatchNormOpDef;
class BatchNormOp;
class BatchNormGradientOpDef;
class BatchNormGradientOp;
class BatchNormGradientofDataOpDef;
class BatchNormGradientofDataOp;
class BatchNormGradientofScaleOpDef;
class BatchNormGradientofScaleOp;
class BatchNormGradientofBiasOpDef;
class BatchNormGradientofBiasOp;

class BatchNormOpDef : public OperatorDef {
 private:
  friend class BatchNormOp;
  struct constrcutor_access_key {};

 public:
  BatchNormOpDef(const constrcutor_access_key&, Tensor input, Tensor bn_scale,
                 Tensor bn_bias, double momentum = 0.1, double eps = 1e-5,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormOp), {input, bn_scale, bn_bias}, op_meta),
    _momentum(momentum),
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

  NDArray running_mean;

  NDArray running_var;

  NDArray save_mean;

  NDArray save_var;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  double _momentum;

  double _eps;
};

class BatchNormOp final : public OpWrapper<BatchNormOpDef> {
 public:
  BatchNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
              double momentum = 0.1, double eps = 1e-5,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormOpDef>(
      make_ptr<BatchNormOpDef>(BatchNormOpDef::constrcutor_access_key(), input,
                               bn_scale, bn_bias, momentum, eps, op_meta)) {}
};

class BatchNormGradientOpDef : public OperatorDef {
 private:
  friend class BatchNormGradientOp;
  struct constrcutor_access_key {};

 public:
  BatchNormGradientOpDef(const constrcutor_access_key&, Tensor output_grad,
                         Tensor input, Tensor bn_scale,
                         BatchNormOp forward_node, double eps,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormGradientOp), {output_grad, input, bn_scale},
                op_meta),
    _forward_node(forward_node),
    _eps(eps) {
    AddOutput(output_grad->meta());
    DeduceStates();
  }

  void DeduceStates() override;

  BatchNormOp get_forward_node() const {
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

  BatchNormOp _forward_node;

  double _eps;
};

class BatchNormGradientOp final : public OpWrapper<BatchNormGradientOpDef> {
 public:
  BatchNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                      BatchNormOp forward_node, double eps,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormGradientOpDef>(make_ptr<BatchNormGradientOpDef>(
      BatchNormGradientOpDef::constrcutor_access_key(), output_grad, input,
      bn_scale, forward_node, eps, op_meta)) {}
};

/*————————————BatchNormGradientofDataOp————————————————————*/
class BatchNormGradientofDataOpDef : public OperatorDef {
 private:
  friend class BatchNormGradientofDataOp;
  struct constrcutor_access_key {};

 public:
  BatchNormGradientofDataOpDef(const constrcutor_access_key&,
                               Tensor bn_gradient, Tensor input,
                               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormGradientofDataOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
    DeduceStates();
  }

  void DeduceStates() override;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BatchNormGradientofDataOp final
: public OpWrapper<BatchNormGradientofDataOpDef> {
 public:
  BatchNormGradientofDataOp(Tensor bn_gradient, Tensor input,
                            const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormGradientofDataOpDef>(
      make_ptr<BatchNormGradientofDataOpDef>(
        BatchNormGradientofDataOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

/*————————————BatchNormGradientofScaleOp————————————————————*/
class BatchNormGradientofScaleOpDef : public OperatorDef {
 private:
  friend class BatchNormGradientofScaleOp;
  struct constrcutor_access_key {};

 public:
  BatchNormGradientofScaleOpDef(const constrcutor_access_key&,
                                Tensor bn_gradient, Tensor input,
                                const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormGradientofScaleOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
    DeduceStates();
  }

  void DeduceStates() override;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BatchNormGradientofScaleOp final
: public OpWrapper<BatchNormGradientofScaleOpDef> {
 public:
  BatchNormGradientofScaleOp(Tensor bn_gradient, Tensor input,
                             const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormGradientofScaleOpDef>(
      make_ptr<BatchNormGradientofScaleOpDef>(
        BatchNormGradientofScaleOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

/*————————————BatchNormGradientofBiasOp————————————————————*/
class BatchNormGradientofBiasOpDef : public OperatorDef {
 private:
  friend class BatchNormGradientofBiasOp;
  struct constrcutor_access_key {};

 public:
  BatchNormGradientofBiasOpDef(const constrcutor_access_key&,
                               Tensor bn_gradient, Tensor input,
                               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchNormGradientofBiasOp), {bn_gradient, input},
                op_meta) {
    AddOutput(bn_gradient->meta());
    DeduceStates();
  }

  void DeduceStates() override; // 有待确认?

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BatchNormGradientofBiasOp final
: public OpWrapper<BatchNormGradientofBiasOpDef> {
 public:
  BatchNormGradientofBiasOp(Tensor bn_gradient, Tensor input,
                            const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchNormGradientofBiasOpDef>(
      make_ptr<BatchNormGradientofBiasOpDef>(
        BatchNormGradientofBiasOpDef::constrcutor_access_key(), bn_gradient,
        input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
