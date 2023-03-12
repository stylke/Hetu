#pragma once

#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

class Optimizer {
 public:
  Optimizer(float learning_rate) : _learning_rate(learning_rate) {}

  Optimizer(TensorList params, float learning_rate)
  : _params(std::move(params)), _learning_rate(learning_rate) {
    HT_VALUE_ERROR_IF(_params.empty()) << "No parameters are provided";
    for (auto& param : _params) {
      HT_VALUE_ERROR_IF(!param->is_parameter())
        << "Tensor " << param << " is not a parameter";
    }
  }

  Tensor Minimize(const Tensor& loss, const TensorList& var_list = {},
                  const Tensor& grad_loss = {}, const OpName& name = "");

  virtual GradAndVarList ComputeGradients(const Tensor& loss,
                                          const TensorList& var_list = {},
                                          const Tensor& grad_loss = {});

  virtual Tensor ApplyGradients(const GradAndVarList& grads_and_vars,
                                const OpName& name = OpName());

  float learning_rate() const {
    return _learning_rate;
  }

 protected:
  virtual Tensor ApplyDense(const GradAndVar& grad_and_var) = 0;

  virtual Tensor MakeStates(const Tensor& variable, const OpName& state_name);

  TensorList _params;
  float _learning_rate;
};

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(float learning_rate, float momentum = 0.9f,
               bool nesterov = false)
  : Optimizer(learning_rate) {
    _init(momentum, nesterov);
  }

  SGDOptimizer(TensorList params, float learning_rate, float momentum = 0.9f,
               bool nesterov = false)
  : Optimizer(std::move(params), learning_rate) {
    _init(momentum, nesterov);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var);

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  void _init(float momentum, bool nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
    _momentum = momentum;
    _nesterov = nesterov;
  }

  float _momentum;
  bool _nesterov;
};

} // namespace graph
} // namespace hetu
