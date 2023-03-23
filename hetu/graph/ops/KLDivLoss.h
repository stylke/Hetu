#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class KLDivLossOpImpl;
class KLDivLossOp;
class KLDivLossGradientOpImpl;
class KLDivLossGradientOp;

class KLDivLossOpImpl : public OpInterface {

 public:
  KLDivLossOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(KLDivLossOp)),
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
    NDArrayMeta output_meta;
    if (_reduction != kNONE)
      output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype()).set_shape({1}).set_device(inputs[0]->device());
    else
      output_meta = inputs[0]->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  ReductionType _reduction;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const KLDivLossOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       const OpMeta& op_meta = OpMeta());

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       const OpMeta& op_meta = OpMeta());

class KLDivLossGradientOpImpl : public OpInterface {

 public:
  KLDivLossGradientOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(KLDivLossGradientOp)),
    _reduction(reduction) {
  }

  ReductionType reduction() const {
    return _reduction;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  ReductionType _reduction;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const KLDivLossOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeKLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
