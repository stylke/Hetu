#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SoftshrinkOpImpl;
class SoftshrinkOp;
class SoftshrinkGradientOpImpl;
class SoftshrinkGradientOp;

class SoftshrinkOpImpl final : public UnaryOpImpl {
 private:
  friend class SoftshrinkOp;
  struct constrcutor_access_key {};

 public:
  SoftshrinkOpImpl(double lambda)
  : UnaryOpImpl(quote(SoftshrinkOp)), _lambda(lambda){
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
      const auto& rhs_ = reinterpret_cast<const SoftshrinkOpImpl&>(rhs);
      if (lambda() == rhs_.lambda())
        return true;
    }
    return false; 
  }

 protected:
  double _lambda;
};

Tensor MakeSoftshrinkOp(Tensor input, double lambda, OpMeta op_meta = OpMeta());

class SoftshrinkGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  SoftshrinkGradientOpImpl(double lambda)
  : UnaryGradientOpImpl(quote(SoftshrinkGradientOp)), _lambda(lambda){
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
      const auto& rhs_ = reinterpret_cast<const SoftshrinkGradientOpImpl&>(rhs);
      if (lambda() == rhs_.lambda())
        return true;
    }
    return false;  
  }

 protected:
  double _lambda;
};

Tensor MakeSoftshrinkGradientOp(Tensor output, Tensor grad_output,
                                double lambda, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
