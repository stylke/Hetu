#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SoftmaxCrossEntropyOpDef;
class SoftmaxCrossEntropyOp;
class SoftmaxCrossEntropyGradientOpDef;
class SoftmaxCrossEntropyGradientOp;

class SoftmaxCrossEntropyOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropyOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropyOpDef(const constrcutor_access_key&, Tensor preds, 
                           Tensor labels, bool reduce = true,
                           const std::string& reduction = "mean",
                           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropyOp), {preds, labels}, op_meta),
    _reduce(reduce),
    _reduction(reduction) {
    HTShape output_shape = {};
    output_shape.reserve(preds->ndim());
    for (size_t i = 0; i < preds->ndim() - 1; ++i) {
      output_shape.emplace_back(preds->shape(i));
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(-1);
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
  }

  bool reduce() const { return _reduce; }

  const std::string& reduction() const { return _reduction; }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _reduce;

  std::string _reduction;
};

class SoftmaxCrossEntropyOp final : public OpWrapper<SoftmaxCrossEntropyOpDef> {
 public:
  SoftmaxCrossEntropyOp(Tensor preds, Tensor labels, bool reduce = true,
                        const std::string& reduction = "mean",
                        const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyOpDef>(make_ptr<SoftmaxCrossEntropyOpDef>(
      SoftmaxCrossEntropyOpDef::constrcutor_access_key(), preds, labels, reduce, reduction,
      op_meta)) {}
};

class SoftmaxCrossEntropyGradientOpDef : public OperatorDef {
 private:
  friend class SoftmaxCrossEntropyGradientOp;
  struct constrcutor_access_key {};

 public:
  SoftmaxCrossEntropyGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                   Tensor labels, Tensor grad_output, bool reduce = true,
                                   const std::string& reduction = "mean",
                                   const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SoftmaxCrossEntropyGradientOp),
                {preds, labels, grad_output}, op_meta),
    _reduce(reduce),
    _reduction(reduction) {
    AddOutput(preds->meta());
  }

  bool reduce() const { return _reduce; }

  const std::string& reduction() const { return _reduction; }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _reduce;

  std::string _reduction;
};

class SoftmaxCrossEntropyGradientOp final
: public OpWrapper<SoftmaxCrossEntropyGradientOpDef> {
 public:
  SoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output, bool reduce = true,
                                const std::string& reduction = "mean",
                                const OpMeta& op_meta = OpMeta())
  : OpWrapper<SoftmaxCrossEntropyGradientOpDef>(
      make_ptr<SoftmaxCrossEntropyGradientOpDef>(
        SoftmaxCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduce, reduction, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
