#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SumOpImpl final : public OpInterface {
 public:
  SumOpImpl() : OpInterface(quote(SumOp)) {}

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_VALUE_ERROR_IF(inputs.empty()) << "No inputs are provided";
    // TODO: support broadcast
    int len = inputs.size();
    HTShape output_shape = inputs[0]->shape();
    for (int i = 1; i < len; ++i) {
      if (inputs[i]->has_shape())
        output_shape = Broadcast(output_shape, inputs[i]->shape());
    }
    auto output_meta = NDArrayMeta(output_shape, inputs[0]->dtype(), inputs[0]->device());
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    TensorList grad_inputs;
    grad_inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++)
      grad_inputs.push_back(op->require_grad(i) ? grad_outputs.front() : Tensor());
    return grad_inputs;
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    // TODO: support broadcast
    int len = input_shapes.size();
    HTShape output_shape = input_shapes[0];
    for (int i = 1; i < len; ++i) {
      output_shape = Broadcast(output_shape, input_shapes[i]);
    }
    return {output_shape};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    auto stream_id = op->instantiation_ctx().stream_index;
    auto& output = outputs.front();
    NDArray::zeros_(output, stream_id);
    for (auto& input : inputs) {
      NDArray::add(input, output, stream_id, output);
    }
  }
};

Tensor MakeSumOp(TensorList inputs, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
