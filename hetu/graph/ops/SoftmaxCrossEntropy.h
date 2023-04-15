#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SoftmaxCrossEntropyOpImpl;
class SoftmaxCrossEntropyOp;
class SoftmaxCrossEntropyGradientOpImpl;
class SoftmaxCrossEntropyGradientOp;

class SoftmaxCrossEntropyOpImpl : public OpInterface {

 public:
  SoftmaxCrossEntropyOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(SoftmaxCrossEntropyOp)),
    _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape output_shape = {};
    for (size_t i = 0; i < inputs[0]->ndim() - 1; ++i) {
      output_shape.emplace_back(inputs[0]->shape(i));
    }
    NDArrayMeta out_meta = inputs[0]->meta();
    if (_reduction != kNONE)
      out_meta.set_shape({1});
    else
      out_meta.set_shape(output_shape);
    return {out_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropyOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 ReductionType reduction = kMEAN,
                                 OpMeta op_meta = OpMeta());

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 const std::string& reduction = "mean",
                                 OpMeta op_meta = OpMeta());

class SoftmaxCrossEntropyGradientOpImpl : public OpInterface {
 public:
  SoftmaxCrossEntropyGradientOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(SoftmaxCrossEntropyGradientOp)),
    _reduction(reduction) {
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropyGradientOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                         ReductionType reduction = kMEAN,
                                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
