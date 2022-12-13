#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class BinaryCrossEntropyOpDef;
class BinaryCrossEntropyOp;
class BinaryCrossEntropyGradientOpDef;
class BinaryCrossEntropyGradientOp;

class BinaryCrossEntropyOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, bool reduce = true,
                          const std::string& reduction = "mean",  
                          const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyOp), {preds, labels}, op_meta) ,
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

class BinaryCrossEntropyOp final : public OpWrapper<BinaryCrossEntropyOpDef> {
 public:
  BinaryCrossEntropyOp(Tensor preds, Tensor labels, bool reduce = true,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyOpDef>(make_ptr<BinaryCrossEntropyOpDef>(
      BinaryCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      reduce, reduction, op_meta)) {}
};

class BinaryCrossEntropyGradientOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyGradientOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output, bool reduce = true,
                                  const std::string& reduction = "mean",
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyGradientOp),
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

class BinaryCrossEntropyGradientOp final
: public OpWrapper<BinaryCrossEntropyGradientOpDef> {
 public:
  BinaryCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output, bool reduce = true,
                               const std::string& reduction = "mean",
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyGradientOpDef>(
      make_ptr<BinaryCrossEntropyGradientOpDef>(
        BinaryCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, reduce, reduction, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
