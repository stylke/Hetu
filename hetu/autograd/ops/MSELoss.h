#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class MSELossOpDef;
class MSELossOp;
class MSELossGradientOpDef;
class MSELossGradientOp;

class MSELossOpDef : public OperatorDef {
 private:
  friend class MSELossOp;
  struct constrcutor_access_key {};

 public:
  MSELossOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, bool reduce = true,
                          const std::string& reduction = "mean",  
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MSELossOp), {preds, labels}, op_meta) ,
    _reduce(reduce),
    _reduction(reduction){
    AddOutput(preds->meta());
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

class MSELossOp final : public OpWrapper<MSELossOpDef> {
 public:
  MSELossOp(Tensor preds, Tensor labels, bool reduce = true,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossOpDef>(make_ptr<MSELossOpDef>(
      MSELossOpDef::constrcutor_access_key(), preds, labels,
      reduce, reduction, op_meta)) {}
};

class MSELossGradientOpDef : public OperatorDef {
 private:
  friend class MSELossGradientOp;
  struct constrcutor_access_key {};

 public:
  MSELossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output, bool reduce = true,
                                  const std::string& reduction = "mean",
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MSELossGradientOp),
                {preds, labels, grad_output}, op_meta) {
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

class MSELossGradientOp final
: public OpWrapper<MSELossGradientOpDef> {
 public:
  MSELossGradientOp(Tensor preds, Tensor labels, Tensor grad_output, bool reduce = true,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<MSELossGradientOpDef>(
      make_ptr<MSELossGradientOpDef>(
        MSELossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduce, reduction, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
