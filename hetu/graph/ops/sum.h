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
    for (size_t i = 1; i < inputs.size(); i++) {
      HT_NOT_IMPLEMENTED_IF(inputs[0]->meta().shape != inputs[i]->meta().shape)
        << "Broadcast is not implemented in " << type();
    }
    return {inputs.front()->meta()};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    TensorList grad_inputs;
    grad_inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++)
      grad_inputs.push_back(grad_outputs.front());
    return grad_inputs;
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    // TODO: support broadcast
    for (size_t i = 1; i < input_shapes.size(); i++) {
      HT_NOT_IMPLEMENTED_IF(input_shapes[0] != input_shapes[i])
        << "Broadcast is not implemented in " << type();
    }
    return {input_shapes.front()};
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
