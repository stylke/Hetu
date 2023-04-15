#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LeakyReluOpImpl;
class LeakyReluOp;
class LeakyReluGradientOpImpl;
class LeakyReluGradientOp;

class LeakyReluOpImpl : public OpInterface {
 private:
  friend class LeakyReluOp;
  struct constrcutor_access_key {};

 public:
  LeakyReluOpImpl(double alpha)
  : OpInterface(quote(LeakyReluOp)), _alpha(alpha) {
  }

  double get_alpha() const {
    return _alpha;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _alpha;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LeakyReluOpImpl&>(rhs);
      return (get_alpha() == rhs_.get_alpha()); 
    }
    return false;
  }
};

Tensor MakeLeakyReluOp(Tensor input, double alpha, OpMeta op_meta = OpMeta());

class LeakyReluGradientOpImpl : public OpInterface {
 private:
  friend class LeakyReluGradientOp;
  struct constrcutor_access_key {};

 public:
  LeakyReluGradientOpImpl(double alpha,
                          OpMeta op_meta = OpMeta())
  : OpInterface(quote(LeakyReluGradientOp)),
    _alpha(alpha) {
  }

  double get_alpha() const {
    return _alpha;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _alpha;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LeakyReluOpImpl&>(rhs);
      return (get_alpha() == rhs_.get_alpha()); 
    }
    return false;
  }
};

Tensor MakeLeakyReluGradientOp(Tensor input, Tensor grad_output, double alpha,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
 