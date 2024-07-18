#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class HardshrinkOpImpl;
class HardshrinkOp;
class HardshrinkGradientOpImpl;
class HardshrinkGradientOp;

class HardshrinkOpImpl final : public UnaryOpImpl {
 private:
  friend class HardshrinkOp;
  struct constructor_access_key {};

 public:
  HardshrinkOpImpl(double lambda)
  : UnaryOpImpl(quote(HardshrinkOp)), _lambda(lambda){
  }

  double lambda() const {
    return _lambda;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const HardshrinkOpImpl&>(rhs);
      if (lambda() == rhs_.lambda())
        return true;
    }
    return false; 
  }

 protected:
  double _lambda;
};

Tensor MakeHardshrinkOp(Tensor input, double lambda, OpMeta op_meta = OpMeta());

class HardshrinkGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  HardshrinkGradientOpImpl(double lambda)
  : UnaryGradientOpImpl(quote(HardshrinkGradientOp)), _lambda(lambda){
  }

  double lambda() const {
    return _lambda;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const HardshrinkGradientOpImpl&>(rhs);
      if (lambda() == rhs_.lambda())
        return true;
    }
    return false;  
  }

 protected:
  double _lambda;
};

Tensor MakeHardshrinkGradientOp(Tensor output, Tensor grad_output,
                                double lambda, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
