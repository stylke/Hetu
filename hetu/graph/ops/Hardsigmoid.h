#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class HardsigmoidOpImpl;
class HardsigmoidOp;
class HardsigmoidGradientOpImpl;
class HardsigmoidGradientOp;

class HardsigmoidOpImpl final : public UnaryOpImpl {
 private:
  friend class HardsigmoidOp;
  struct constructor_access_key {};

 public:
  HardsigmoidOpImpl()
  : UnaryOpImpl(quote(HardsigmoidOp)){
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

Tensor MakeHardsigmoidOp(Tensor input, OpMeta op_meta = OpMeta());

class HardsigmoidGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  HardsigmoidGradientOpImpl()
  : UnaryGradientOpImpl(quote(HardsigmoidGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeHardsigmoidGradientOp(Tensor output, Tensor grad_output,
                                 OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
