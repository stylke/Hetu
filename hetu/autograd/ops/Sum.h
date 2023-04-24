#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SumOpDef;
class SumOp;

class SumOpDef : public OperatorDef {
 private:
  friend class SumOp;
  struct constrcutor_access_key {};

 public:
  SumOpDef(const constrcutor_access_key&, TensorList inputs,
           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SumOp), inputs, op_meta) {
    HT_ASSERT(_inputs.size() > 0) << "No inputs are provided";
    AddOutput(_inputs[0]->meta());
    if (op_meta.is_deduce_states) {  
      DeduceStates();
    }
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class SumOp final : public OpWrapper<SumOpDef> {
 public:
  SumOp(TensorList inputs, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SumOpDef>(make_ptr<SumOpDef>(SumOpDef::constrcutor_access_key(),
                                           inputs, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
