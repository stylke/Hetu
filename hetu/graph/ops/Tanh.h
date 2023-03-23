#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class TanhOpImpl;
class TanhOp;
class TanhGradientOpImpl;
class TanhGradientOp;

class TanhOpImpl : public OpInterface {

 public:
  TanhOpImpl()
  : OpInterface(quote(TanhOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
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

Tensor MakeTanhOp(Tensor input, const OpMeta& op_meta = OpMeta());

class TanhGradientOpImpl : public OpInterface {

 public:
  TanhGradientOpImpl()
  : OpInterface(quote(TanhGradientOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;
 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeTanhGradientOp(Tensor input, Tensor grad_output,
                          const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
