#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class NLLLossOpImpl;
class NLLLossOp;
class NLLLossGradientOpImpl;
class NLLLossGradientOp;

class NLLLossOpImpl : public OpInterface {
 public:
  NLLLossOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(NLLLossOp)),
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
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      HT_ASSERT(inputs[0]->ndim() == inputs[1]->ndim() + 1)
      << "Input's dims should be 1 more than label's."
      << "Input dims: " << inputs[0]->ndim()
      << ". Label dims: " << inputs[1]->ndim();
      for (size_t i = 0; i < inputs[1]->ndim(); i++)
        HT_ASSERT(inputs[0]->shape(i) == inputs[1]->shape(i));
    }
    NDArrayMeta output_meta;
    if (_reduction != kNONE)
      output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype()).set_shape({1}).set_device(inputs[0]->device());
    else
      output_meta = inputs[1]->meta();
    return {output_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const NLLLossOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeNLLLossOp(Tensor preds, Tensor labels,
                     ReductionType reduction = kMEAN,
                     OpMeta op_meta = OpMeta());

Tensor MakeNLLLossOp(Tensor preds, Tensor labels,
                     const std::string& reduction = "mean",
                     OpMeta op_meta = OpMeta());

class NLLLossGradientOpImpl : public OpInterface {
 public:
  NLLLossGradientOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(NLLLossGradientOp)),
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
      const auto& rhs_ = reinterpret_cast<const NLLLossGradientOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeNLLLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                             ReductionType reduction = kMEAN,
                             OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
