#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class KLDivLossOpDef;
class KLDivLossOp;
class KLDivLossGradientOpDef;
class KLDivLossGradientOp;

class KLDivLossOpDef : public OperatorDef {
 private:
  friend class KLDivLossOp;
  struct constrcutor_access_key {};

 public:
  KLDivLossOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, bool reduce = true,
                          const std::string& reduction = "mean",  
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(KLDivLossOp), {preds, labels}, op_meta) ,
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

class KLDivLossOp final : public OpWrapper<KLDivLossOpDef> {
 public:
  KLDivLossOp(Tensor preds, Tensor labels, bool reduce = true,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossOpDef>(make_ptr<KLDivLossOpDef>(
      KLDivLossOpDef::constrcutor_access_key(), preds, labels,
      reduce, reduction, op_meta)) {}
};

class KLDivLossGradientOpDef : public OperatorDef {
 private:
  friend class KLDivLossGradientOp;
  struct constrcutor_access_key {};

 public:
  KLDivLossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output, bool reduce = true,
                                  const std::string& reduction = "mean",
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(KLDivLossGradientOp),
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

class KLDivLossGradientOp final
: public OpWrapper<KLDivLossGradientOpDef> {
 public:
  KLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output, bool reduce = true,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<KLDivLossGradientOpDef>(
      make_ptr<KLDivLossGradientOpDef>(
        KLDivLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduce, reduction, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
