#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class BoolOpImpl;
class BoolOp;

class BoolOpImpl : public OpInterface {

 public:
  BoolOpImpl()
  : OpInterface(quote(BoolOp)) {
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  
 public:
  bool operator==(const OpInterface& rhs) const override {
    return (OpInterface::operator==(rhs));
  }
};

Tensor MakeBoolOp(Tensor input, const OpMeta& op_meta = OpMeta());


} // namespace autograd
} // namespace hetu
