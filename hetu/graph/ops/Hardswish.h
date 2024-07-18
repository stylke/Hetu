#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class HardswishOpImpl;
class HardswishOp;
class HardswishGradientOpImpl;
class HardswishGradientOp;

class HardswishOpImpl final : public UnaryOpImpl {
 private:
  friend class HardswishOp;
  struct constructor_access_key {};

 public:
  HardswishOpImpl()
  : UnaryOpImpl(quote(HardswishOp)){
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs); 
  }
};

Tensor MakeHardswishOp(Tensor input, OpMeta op_meta = OpMeta());

class HardswishGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  HardswishGradientOpImpl()
  : UnaryGradientOpImpl(quote(HardswishGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeHardswishGradientOp(Tensor input, Tensor grad_output,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
