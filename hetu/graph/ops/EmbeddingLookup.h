#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class EmbeddingLookupOpImpl;
class EmbeddingLookupOp;
class EmbeddingLookupGradientOpImpl;
class EmbeddingLookupGradientOp;

class EmbeddingLookupOpImpl : public OpInterface {
 public:
  EmbeddingLookupOpImpl()
  : OpInterface(quote(EmbeddingLookupOp)) {
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape;
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      shape = inputs[1]->shape();
      shape.emplace_back(inputs[0]->shape(1));
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeEmbeddingLookupOp(Tensor input, Tensor id, const OpMeta& op_meta = OpMeta());

class EmbeddingLookupGradientOpImpl : public OpInterface {
 public:
  EmbeddingLookupGradientOpImpl(const OpMeta& op_meta = OpMeta())
  : OpInterface(quote(EmbeddingLookupGradientOp)) {
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs[3]->meta();
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }

};

Tensor MakeEmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input, Tensor input,
                                     const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
