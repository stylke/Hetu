#include "hetu/graph/optim/optimizer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/optimizer_update.h"

namespace hetu {
namespace graph {

Tensor Optimizer::Minimize(const Tensor& loss, const TensorList& var_list,
                           const Tensor& grad_loss, const OpName& name) {
  GradAndVarList grads_and_vars = ComputeGradients(loss, var_list, grad_loss);
  GradAndVarList filtered_grads_and_vars;
  filtered_grads_and_vars.reserve(grads_and_vars.size());
  std::copy_if(grads_and_vars.begin(), grads_and_vars.end(),
               std::back_inserter(filtered_grads_and_vars),
               [](const GradAndVar& n) { return n.first.is_defined(); });
  return ApplyGradients(filtered_grads_and_vars, name);
}

Tensor Optimizer::ApplyGradients(const GradAndVarList& grads_and_vars,
                                 const OpName& name, const Tensor& infinite_count) {
  TensorList updated_params;
  updated_params.reserve(grads_and_vars.size());
  std::transform(
    grads_and_vars.begin(), grads_and_vars.end(),
    std::back_inserter(updated_params),
    [&](const GradAndVar& grad_and_var) { return ApplyDense(grad_and_var, infinite_count); });
  return MakeGroupOp(OpMeta().set_extra_deps(updated_params).set_name(name));
}

Tensor Optimizer::MakeStates(const Tensor& variable, const OpName& state_name) {
  const auto& producer = variable->producer();
  HT_VALUE_ERROR_IF(!producer->is_parameter());
  // special case: Varibale States should be set distributed_states
  const DistributedStates& ds_variable = variable->get_distributed_states(); 
  HT_ASSERT (ds_variable.is_valid()) 
    << "Diastributed States for varibale " << variable << " must be valid!";  
  Tensor states = MakeVariableOp(ZerosInitializer(), variable->shape(),
                                 variable->dtype(), false, ds_variable, 
                                 OpMeta()
                                   .set_device_group(producer->device_group())
                                   .set_eager_device(producer->eager_device())
                                   .set_name(variable->name() + "_" + state_name));

  return std::move(states);
}

GradAndVarList Optimizer::ComputeGradients(const Tensor& loss,
                                           const TensorList& var_list,
                                           const Tensor& grad_loss) {
  TensorList vars = var_list;
  if (vars.empty()) {
    auto topo_order = Graph::TopoSort(loss);
    for (auto& op_ref : topo_order)
      if (op_ref.get()->is_parameter())
        vars.push_back(op_ref.get()->output(0));
  }
  TensorList grads = Graph::Gradients(loss, vars, grad_loss);
  GradAndVarList grads_and_vars;
  grads_and_vars.reserve(grads.size());
  for (size_t i = 0; i < grads.size(); i++)
    grads_and_vars.emplace_back(grads[i], vars[i]);
  return grads_and_vars;
}

Tensor SGDOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group(var->producer()->device_group())
                          .set_name("Update_" + var->name());
  if (momentum() == 0) {
    if (infinite_count != Tensor())
      return MakeSGDUpdateWithGradScalerOp(var, grad, infinite_count, learning_rate(), update_op_meta);
    return MakeSGDUpdateOp(var, grad, learning_rate(), update_op_meta);
  } else {
    return MakeMomentumUpdateOp(var, grad, MakeStates(var, "velocity"),
                                learning_rate(), momentum(), nesterov(),
                                update_op_meta);
  }
}

Tensor AdamOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group(var->producer()->device_group())
                          .set_name("Update_" + var->name());
  NDArray step = NDArray::ones({1}, kCPU, kInt64, kBlockingStream);
  std::vector<Device> cpus = {kCPU};
  DeviceGroup cpu_group = DeviceGroup(cpus);
  HTShape step_shape = {1};
  Tensor step1 = MakeVariableOp(OnesInitializer(), step_shape, kInt64,
                                false, var->get_distributed_states(), 
                                OpMeta()
                                  .set_device_group(var->producer()->device_group())
                                  .set_eager_device(kCPU)
                                  .set_name(var->name() + "_step"));
  return MakeAdamOp(var, grad, MakeStates(var, "mean"),
                    MakeStates(var, "variance"),
                    learning_rate(), step1, beta1(), beta2(),
                    eps(), weight_decay(), update_op_meta);
}

} // namespace graph
} // namespace hetu