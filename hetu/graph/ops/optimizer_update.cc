#include "hetu/graph/headers.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

bool OptimizerUpdateOpInterface::DoMapToParallelDevices(
  Operator& op, const DeviceGroup& placement_group) const {
  // use comm_op instead
  // if (placement_group.num_devices() > 1) {
  //   // TODO
  //   HT_NOT_IMPLEMENTED << "Fill this up with AllReduceOpImpl";
  // }
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

void AdamOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                           NDArrayList& outputs,
                           RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray& mean = const_cast<NDArray&>(inputs.at(2));
  NDArray& variance = const_cast<NDArray&>(inputs.at(3));
  NDArray& step = const_cast<NDArray&>(inputs.at(4));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::Adam, grad, param,
                                  mean, variance, step, learning_rate(), 
                                  beta1(), beta2(), eps(), weight_decay(), 
                                  op->instantiation_ctx().stream());
}

void AdamOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta) const {
  // default: distributed states of output tensor directly copy from input tensor
  // check input states is valid & check distributed states of all input tensor are the same.
  HT_ASSERT(inputs.size() > 0) << op_meta.name << ": distributed states should be manually set when in_degree=0!";
  HT_LOG_DEBUG << op_meta.name << ": default copy states from inputs";
  DistributedStates default_ds;
  for (int i = 0; i < 4; ++i) {
    auto& input = inputs.at(i);
    const auto& input_ds = input->get_distributed_states(); 
    HT_ASSERT(input_ds.is_valid()) << op_meta.name << ": input states must be valid! and " 
                                    << "input: " << input << ", input_ds: " << input_ds.ds_info();
    HT_ASSERT(input_ds.get_dim(-2) == 1) << op_meta.name << ": input shouldn't be partial!";      
    if (!default_ds.is_valid()) {
      default_ds.set_distributed_states(input_ds);
    } else {
      HT_ASSERT(default_ds.check_equal(input_ds))
        << op_meta.name << ": in Default DoDeduceStates: distributed states of all input tensor must be same!"
        << ", " << default_ds.ds_info() << " vs " << input_ds.ds_info();
    }
  }
  for (auto& output : outputs) {
    output->set_distributed_states(default_ds);
  }
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

Tensor MakeAdamOp(Tensor param, Tensor grad, Tensor mean, Tensor variance,
                  float learning_rate, Tensor step, float beta1, float beta2, 
                  float eps, float weight_decay, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<AdamOpImpl>(
                       learning_rate, beta1, beta2, eps, weight_decay),
                       {std::move(param), std::move(grad), std::move(mean), std::move(variance), std::move(step)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
