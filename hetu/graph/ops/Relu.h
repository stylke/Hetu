#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ReluOpImpl;
class ReluOp;
class ReluGradientOpImpl;
class ReluGradientOp;

class ReluOpImpl : public OpInterface {
 private:
  friend class ReluOp;
  struct constrcutor_access_key {};

 public:
  ReluOpImpl()
  : OpInterface(quote(ReluOp)) {
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

Tensor MakeReluOp(Tensor input, const OpMeta& op_meta = OpMeta());

class ReluGradientOpImpl : public OpInterface {

 public:
  ReluGradientOpImpl()
  : OpInterface(quote(ReluGradientOp)) {
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

Tensor MakeReluGradientOp(Tensor input, Tensor grad_output,
                          const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
