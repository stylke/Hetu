#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SoftplusOpImpl;
class SoftplusOp;
class SoftplusGradientOpImpl;
class SoftplusGradientOp;

class SoftplusOpImpl final : public UnaryOpImpl {
 private:
  friend class SoftplusOp;
  struct constrcutor_access_key {};

 public:
  SoftplusOpImpl(double beta, double threshold)
  : UnaryOpImpl(quote(SoftplusOp)), 
  _beta(beta), _threshold(threshold){
  }

  double beta() const {
    return _beta;
  }

  double threshold() const {
    return _threshold;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftplusOpImpl&>(rhs);
      if (beta() == rhs_.beta()
          && threshold() == rhs_.threshold())
        return true;
    }
    return false; 
  }

 protected:
  double _beta;
  double _threshold;
};

Tensor MakeSoftplusOp(Tensor input, double beta, double threshold, OpMeta op_meta = OpMeta());

class SoftplusGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  SoftplusGradientOpImpl(double beta, double threshold)
  : UnaryGradientOpImpl(quote(SoftplusGradientOp)), 
  _beta(beta), _threshold(threshold){
  }

  double beta() const {
    return _beta;
  }

  double threshold() const {
    return _threshold;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftplusGradientOpImpl&>(rhs);
      if (beta() == rhs_.beta()
          && threshold() == rhs_.threshold())
        return true;
    }
    return false;  
  }

 protected:
  double _beta;
  double _threshold;
};

Tensor MakeSoftplusGradientOp(Tensor input, Tensor grad_output,
                              double beta, double threshold, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
