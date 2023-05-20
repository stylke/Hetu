#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class BinaryCrossEntropyOpImpl;
class BinaryCrossEntropyGradientOpImpl;

class BinaryCrossEntropyOpImpl final : public OpInterface {
 public:
  BinaryCrossEntropyOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(BinaryCrossEntropyOp)), _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[0], inputs[1]);
    NDArrayMeta output_meta = inputs.front()->meta();
    if (reduction() != kNONE)
      output_meta.shape =
        NDArrayMeta::Reduce(output_meta.shape, HTAxes(), false);
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    HTShape output_shape = input_shapes.front();
    if (reduction() != kNONE)
      output_shape = NDArrayMeta::Reduce(output_shape, HTAxes(), false);
    return {output_shape};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BinaryCrossEntropyOpImpl&>(rhs);
      return reduction() == rhs_.reduction();
    }
    return false;
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  ReductionType _reduction;
};

class BinaryCrossEntropyGradientOpImpl final : public OpInterface {
 public:
  BinaryCrossEntropyGradientOpImpl(ReductionType reduction = kMEAN)
  : OpInterface(quote(BinaryCrossEntropyGradientOp)), _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[0], inputs[1]);
    return {inputs[0]->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    HT_ASSERT_GE(input_shapes.at(0).size(), 2)
      << "Invalid shape for " << type() << ": " << input_shapes.at(0);
    return {input_shapes.at(0)};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ =
        reinterpret_cast<const BinaryCrossEntropyGradientOpImpl&>(rhs);
      return reduction() == rhs_.reduction();
    }
    return false;
  }

  ReductionType reduction() const {
    return _reduction;
  }

 protected:
  ReductionType _reduction;
};

Tensor MakeBinaryCrossEntropyOp(Tensor probs, Tensor labels,
                                ReductionType reduction = kMEAN,
                                OpMeta op_meta = OpMeta());

inline Tensor MakeBinaryCrossEntropyOp(Tensor probs, Tensor labels,
                                       std::string reduction,
                                       OpMeta op_meta = OpMeta()) {
  return MakeBinaryCrossEntropyOp(std::move(probs), std::move(labels),
                                  Str2ReductionType(reduction),
                                  std::move(op_meta));
}

Tensor MakeBinaryCrossEntropyGradientOp(Tensor probs, Tensor labels,
                                        Tensor grad_outputs,
                                        ReductionType reduction = kMEAN,
                                        OpMeta op_meta = OpMeta());

inline Tensor MakeBinaryCrossEntropyGradientOp(Tensor probs, Tensor labels,
                                               Tensor grad_outputs,
                                               std::string reduction,
                                               OpMeta op_meta = OpMeta()) {
  return MakeBinaryCrossEntropyGradientOp(
    std::move(probs), std::move(labels), std::move(grad_outputs),
    Str2ReductionType(reduction), std::move(op_meta));
}

} // namespace graph
} // namespace hetu
