#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SoftmaxCrossEntropySparseOpImpl;
class SoftmaxCrossEntropySparseOp;
class SoftmaxCrossEntropySparseGradientOpImpl;
class SoftmaxCrossEntropySparseGradientOp;

class SoftmaxCrossEntropySparseOpImpl : public OpInterface {
 public:
  SoftmaxCrossEntropySparseOpImpl(const int64_t ignored_index = -1, 
                          ReductionType reduction = kMEAN)
  : OpInterface(quote(SoftmaxCrossEntropySparseOp)),
    _ignored_index(ignored_index),
    _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  }

  ReductionType reduction() const {
    return _reduction;
  }

  int64_t ignored_index() const {
    return _ignored_index;
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
    out_meta.set_device(inputs[0]->device());
    return {out_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _ignored_index;

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropySparseOpImpl&>(rhs);
      return (reduction() == rhs_.reduction()
              && ignored_index() == rhs_.ignored_index());
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                                       ReductionType reduction = kMEAN,
                                       OpMeta op_meta = OpMeta());

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                                       const std::string& reduction = "mean",
                                       OpMeta op_meta = OpMeta());

class SoftmaxCrossEntropySparseGradientOpImpl : public OpInterface {

 public:
  SoftmaxCrossEntropySparseGradientOpImpl(const int64_t ignored_index = -1, 
                                          ReductionType reduction = kMEAN)
  : OpInterface(quote(SoftmaxCrossEntropySparseGradientOp)),
    _ignored_index(ignored_index),
    _reduction(reduction) {
  }

  ino64_t ignored_index() const {
    return _ignored_index;
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

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _ignored_index; 

  ReductionType _reduction;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropySparseGradientOpImpl&>(rhs);
      return (reduction() == rhs_.reduction()
              && ignored_index() == rhs_.ignored_index());
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropySparseGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                               const int64_t ignored_index = -1, 
                                               ReductionType reduction = kMEAN,
                                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
