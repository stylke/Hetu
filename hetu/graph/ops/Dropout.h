#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class DropoutOpImpl;
class DropoutOp;
class DropoutGradientOpImpl;
class DropoutGradientOp;
class DropoutGradientWithRecomputationOpImpl;
class DropoutGradientWithRecomputationOp;

class DropoutOpImpl : public OpInterface {
 public:
  DropoutOpImpl(double keep_prob,
                bool recompute = false, bool inplace = false)
  : OpInterface(quote(DropoutOp)),
    _keep_prob(keep_prob),
    _recompute(recompute || inplace),
    _inplace(inplace) {
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool recompute() const {
    return _recompute;
  }

  bool inplace() const{
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
      const auto& rhs_ = reinterpret_cast<const DropoutOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob()
              && recompute() == rhs_.recompute()
              && inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropoutOp(Tensor input, double keep_prob, bool recompute = false,
                     bool inplace = false, OpMeta op_meta = OpMeta());

class DropoutGradientOpImpl : public OpInterface {
 public:
  DropoutGradientOpImpl(double keep_prob)
  : OpInterface(quote(DropoutGradientOp)),
    _keep_prob(keep_prob) {
  }

  double keep_prob() const {
    return _keep_prob;
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

  double _keep_prob;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DropoutGradientOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob());
    }
    return false;
  }
};

Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor output, double keep_prob,
                             OpMeta op_meta = OpMeta());

class DropoutGradientWithRecomputationOpImpl : public OpInterface {
 private:
  friend class DropoutGradientWithRecomputationOp;
  struct constrcutor_access_key {};

 public:
  DropoutGradientWithRecomputationOpImpl(OpId forward_op,
                                         double keep_prob,
                                         bool inplace,
                                         OpMeta op_meta = OpMeta())
  : OpInterface(quote(DropoutGradientWithRecomputationOp)),
    _forward_op(forward_op),
    _keep_prob(keep_prob),
    _inplace(inplace) {
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool inplace() const{
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
      const auto& rhs_ = reinterpret_cast<const DropoutGradientWithRecomputationOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob()
              && _forward_op == rhs_._forward_op
              && inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropoutGradientWithRecomputationOp(Tensor grad_output, OpId forward_op, double keep_prob,
                                              bool inplace, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
