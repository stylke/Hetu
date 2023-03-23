#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class WhereOpImpl;
class WhereOp;

class WhereOpImpl : public OpInterface {
 private:
  friend class WhereOp;
  struct constrcutor_access_key {};

 public:
  WhereOpImpl()
  : OpInterface(quote(WhereOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[1]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeWhereOp(Tensor cond, Tensor inputA, Tensor inputB,
                   const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
