#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class HardtanhOpImpl;
class HardtanhOp;
class HardtanhGradientOpImpl;
class HardtanhGradientOp;

class HardtanhOpImpl final : public UnaryOpImpl {
 private:
  friend class HardtanhOp;
  struct constructor_access_key {};

 public:
  HardtanhOpImpl(double min_val, double max_val)
  : UnaryOpImpl(quote(HardtanhOp)), 
  _min_val(min_val), _max_val(max_val){
  }

  double min_val() const {
    return _min_val;
  }

  double max_val() const {
    return _max_val;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const HardtanhOpImpl&>(rhs);
      if (min_val() == rhs_.min_val()
          && max_val() == rhs_.max_val())
        return true;
    }
    return false; 
  }

 protected:
  double _min_val;
  double _max_val;
};

Tensor MakeHardtanhOp(Tensor input, double min_val, double max_val, OpMeta op_meta = OpMeta());

class HardtanhGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  HardtanhGradientOpImpl(double min_val, double max_val)
  : UnaryGradientOpImpl(quote(HardtanhGradientOp)), 
  _min_val(min_val), _max_val(max_val){
  }

  double min_val() const {
    return _min_val;
  }

  double max_val() const {
    return _max_val;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const HardtanhGradientOpImpl&>(rhs);
      if (min_val() == rhs_.min_val()
          && max_val() == rhs_.max_val())
        return true;
    }
    return false;  
  }

 protected:
  double _min_val;
  double _max_val;
};

Tensor MakeHardtanhGradientOp(Tensor output, Tensor grad_output,
                              double min_val, double max_val, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
