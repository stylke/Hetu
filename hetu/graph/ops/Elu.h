#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class EluOpImpl;
class EluOp;
class EluGradientOpImpl;
class EluGradientOp;

class EluOpImpl final : public UnaryOpImpl {
 private:
  friend class EluOp;
  struct constrcutor_access_key {};

 public:
  EluOpImpl(double alpha, double scale)
  : UnaryOpImpl(quote(EluOp)), 
  _alpha(alpha), _scale(scale){
  }

  double alpha() const {
    return _alpha;
  }

  double scale() const {
    return _scale;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const EluOpImpl&>(rhs);
      if (alpha() == rhs_.alpha()
          && scale() == rhs_.scale())
        return true;
    }
    return false; 
  }

 protected:
  double _alpha;
  double _scale;
};

Tensor MakeEluOp(Tensor input, double alpha, double scale, OpMeta op_meta = OpMeta());

class EluGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  EluGradientOpImpl(double alpha, double scale)
  : UnaryGradientOpImpl(quote(EluGradientOp)), 
  _alpha(alpha), _scale(scale){
  }

  double alpha() const {
    return _alpha;
  }

  double scale() const {
    return _scale;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const EluGradientOpImpl&>(rhs);
      if (alpha() == rhs_.alpha()
          && scale() == rhs_.scale())
        return true;
    }
    return false;  
  }

 protected:
  double _alpha;
  double _scale;
};

Tensor MakeEluGradientOp(Tensor output, Tensor grad_output,
                         double alpha, double scale, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
