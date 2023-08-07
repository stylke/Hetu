#include "hetu/graph/headers.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

bool OptimizerUpdateOpInterface::DoMapToParallelDevices(
  Operator& op, const DeviceGroup& placement_group) const {
  if (placement_group.num_devices() > 1) {
    // TODO
    HT_NOT_IMPLEMENTED << "Fill this up with AllReduceOpImpl";
  }
  return OpInterface::DoMapToParallelDevices(op, placement_group);
}

void SGDUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

void SGDUpdateWithGradScalerOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  const NDArray& infinite_count = inputs.at(2);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(),
                               type(), hetu::impl::SGDUpdateWithGradScaler, grad, infinite_count, 
                               param, velocity, learning_rate(), 0, false,
                               op->instantiation_ctx().stream());
}

void MomentumUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity = inputs.at(2);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, float learning_rate,
                       OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateOpImpl>(learning_rate),
                       {std::move(param), std::move(grad)}, std::move(op_meta))
    ->output(0);
}

Tensor MakeSGDUpdateWithGradScalerOp(Tensor param, Tensor grad, Tensor infinite_count, 
                                     float learning_rate, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateWithGradScalerOpImpl>(learning_rate),
                       {std::move(param), std::move(grad), std::move(infinite_count)}, std::move(op_meta))
    ->output(0);
}


Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            float learning_rate, float momentum, bool nesterov,
                            OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<MomentumUpdateOpImpl>(
                         learning_rate, momentum, nesterov),
                       {std::move(param), std::move(grad), std::move(velocity)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
