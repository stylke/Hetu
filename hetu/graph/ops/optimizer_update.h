#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class OptimizerUpdateOpInterface : public OpInterface {
 public:
  OptimizerUpdateOpInterface(OpType&& op_type, float learning_rate)
  : OpInterface(std::move(op_type)), _learning_rate(learning_rate) {
    HT_VALUE_ERROR_IF(_learning_rate < 0)
      << "Invalid learning rate: " << _learning_rate;
  }

  uint64_t op_indicator() const noexcept override {
    return OPTIMIZER_UPDATE_OP;
  }

  bool inplace_at(size_t input_position) const override {
    // By default, the first input is parameter, the second is gradient,
    // and the rest are optimizer states.
    return input_position != 1;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    // Question: should we check whether the param is trainable?
    HT_VALUE_ERROR_IF(!inputs.front()->producer()->is_parameter())
      << "The first input " << inputs.front() << " is not a parameter";
    return {inputs.front()->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override {
    // In place update
    return {inputs.front()};
  }

  bool
  DoMapToParallelDevices(Operator& op,
                         const DeviceGroup& placement_group) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ =
        reinterpret_cast<const OptimizerUpdateOpInterface&>(rhs);
      return learning_rate() == rhs_.learning_rate();
    }
    return false;
  }

  float learning_rate() const {
    return _learning_rate;
  }

 protected:
  float _learning_rate;
};

class SGDUpdateOpImpl : public OptimizerUpdateOpInterface {
 public:
  SGDUpdateOpImpl(float learning_rate)
  : OptimizerUpdateOpInterface(quote(SGDUpdateOp), learning_rate) {}

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
};

class MomentumUpdateOpImpl : public OptimizerUpdateOpInterface {
 public:
  MomentumUpdateOpImpl(float learning_rate, float momentum, bool nesterov)
  : OptimizerUpdateOpInterface(quote(MomemtumUpdateOp), learning_rate),
    _momentum(momentum),
    _nesterov(nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MomentumUpdateOpImpl&>(rhs);
      return momentum() == rhs_.momentum() && nesterov() == rhs_.nesterov();
    }
    return false;
  }

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  float _momentum;
  bool _nesterov;
};

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, float learning_rate,
                       OpMeta op_meta = OpMeta());

Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            float learning_rate, float momentum, bool nesterov,
                            OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu