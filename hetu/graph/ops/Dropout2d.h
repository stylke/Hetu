#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class Dropout2dOpImpl;
class Dropout2dOp;
class Dropout2dGradientWithRecomputationOpImpl;
class Dropout2dGradientWithRecomputationOp;

class Dropout2dOpImpl : public OpInterface {
 public:
  Dropout2dOpImpl(double keep_prob,
                  bool recompute = false, bool inplace = false)
  : OpInterface(quote(Dropout2dOp)),
    _keep_prob(keep_prob),
    _recompute(recompute || inplace),
    _inplace(inplace) {
    // TODO: support without recomputation
    HT_ASSERT(inplace) << "Currently we require Conv2D to be in place";
  }

  double keep_prob() const {
    return _keep_prob;
  };

  bool recompute() const {
    return _recompute;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  double _keep_prob;
  bool _recompute;
  bool _inplace;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob()
              && recompute() == rhs_.recompute()
              && inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropout2dOp(Tensor input, double keep_prob, bool recompute = false,
                       bool inplace = false, OpMeta op_meta = OpMeta());

class Dropout2dGradientWithRecomputationOpImpl : public OpInterface {
 public:
  Dropout2dGradientWithRecomputationOpImpl(OpId forward_op,
                                           double keep_prob,
                                           bool inplace,
                                           OpMeta op_meta = OpMeta())
  : OpInterface(quote(Dropout2dGradientWithRecomputationOp)),
    _forward_op(forward_op),
    _keep_prob(keep_prob),
    _inplace(inplace) {
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  OpId _forward_op;

  double _keep_prob;

  bool _inplace;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dGradientWithRecomputationOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob()
              && _forward_op == rhs_._forward_op
              && inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropout2dGradientWithRecomputationOp(Tensor grad_output,
                                                OpId forward_op, double keep_prob,
                                                bool inplace,
                                                OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
