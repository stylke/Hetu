#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class NLLLossOpDef;
class NLLLossOp;
class NLLLossGradientOpDef;
class NLLLossGradientOp;

class NLLLossOpDef : public OperatorDef {
 private:
  friend class NLLLossOp;
  struct constrcutor_access_key {};

 public:
  NLLLossOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, bool reduce = true,
                          const std::string& reduction = "mean",  
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NLLLossOp), {preds, labels}, op_meta) ,
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

class NLLLossOp final : public OpWrapper<NLLLossOpDef> {
 public:
  NLLLossOp(Tensor preds, Tensor labels, bool reduce = true,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossOpDef>(make_ptr<NLLLossOpDef>(
      NLLLossOpDef::constrcutor_access_key(), preds, labels,
      reduce, reduction, op_meta)) {}
};

class NLLLossGradientOpDef : public OperatorDef {
 private:
  friend class NLLLossGradientOp;
  struct constrcutor_access_key {};

 public:
  NLLLossGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output, bool reduce = true,
                                  const std::string& reduction = "mean",
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NLLLossGradientOp),
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

class NLLLossGradientOp final
: public OpWrapper<NLLLossGradientOpDef> {
 public:
  NLLLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output, bool reduce = true,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<NLLLossGradientOpDef>(
      make_ptr<NLLLossGradientOpDef>(
        NLLLossGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduce, reduction, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
