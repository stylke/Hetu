#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class MSELossOpImpl;
class MSELossOp;
class MSELossGradientOpImpl;
class MSELossGradientOp;

class MSELossOpImpl : public OpInterface {
 public:
  MSELossOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(MSELossOp)),
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
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MSELossOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeMSELossOp(Tensor preds, Tensor labels,
                     ReductionType reduction = kMEAN,
                     const OpMeta& op_meta = OpMeta());

Tensor MakeMSELossOp(Tensor preds, Tensor labels,
                     const std::string& reduction = "mean",
                     const OpMeta& op_meta = OpMeta());

class MSELossGradientOpImpl : public OpInterface {
 public:
  MSELossGradientOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(MSELossGradientOp)),
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
      const auto& rhs_ = reinterpret_cast<const MSELossOpImpl&>(rhs);
      return (reduction() == rhs_.reduction());
    }
    return false;
  }
};

Tensor MakeMSELossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                             ReductionType reduction = kMEAN,
                             const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
