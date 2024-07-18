#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class MishOpImpl;
class MishOp;
class MishGradientOpImpl;
class MishGradientOp;

class MishOpImpl final : public UnaryOpImpl {
 private:
  friend class MishOp;
  struct constructor_access_key {};

 public:
  MishOpImpl()
  : UnaryOpImpl(quote(MishOp)){
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

Tensor MakeMishOp(Tensor input, OpMeta op_meta = OpMeta());

class MishGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  MishGradientOpImpl()
  : UnaryGradientOpImpl(quote(MishGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeMishGradientOp(Tensor input, Tensor grad_output,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
