#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/core/symbol.h"

namespace hetu {
namespace graph {

using namespace hetu::impl::comm;

uint64_t CommOpImpl::get_comm_type(Operator& op) {
  // input may be inplaced, so comm_type should be updated for each call
  Tensor& input = op->input(0);
  auto& graph = input->graph();
  const auto& src_ds = input->get_distributed_states();
  const auto& dst_ds = get_dst_distributed_states(op);
  // 1. pp 2. tp 3. tp+pp 
  if (src_ds.check_equal(dst_ds)) {
    _comm_type = P2P_OP; // P2P OP: in define_and_run_graph: unuse comm op(p2p to self); in exec_graph: real p2p op
    HT_LOG_DEBUG << "P2P_OP"; // need to carefully check in exec_graph!!!
  } 
  // tp (now also included tp+pp case, simplely add p2p op after tp result)
  else if (src_ds.check_pure_duplicate()) {
    // TODO: check data among comm_devices be duplicate
    _comm_type = COMM_SPLIT_OP;
    HT_LOG_DEBUG << "COMM_SPLIT_OP";
  } else if (src_ds.check_allreduce(dst_ds)) {
    _comm_type = ALL_REDUCE_OP;
    HT_LOG_DEBUG << "ALL_REDUCE_OP";
  } else if (src_ds.check_allgather(dst_ds)) {
    _comm_type = ALL_GATHER_OP;
    HT_LOG_DEBUG << "ALL_GATHER_OP";
  } else if (src_ds.check_reducescatter(dst_ds)) {
    _comm_type = REDUCE_SCATTER_OP;
    HT_LOG_DEBUG << "REDUCE_SCATTER_OP";
  } else {
    _comm_type = BATCHED_ISEND_IRECV_OP;
    HT_LOG_DEBUG << "BATCHED_ISEND_IRECV_OP";
  }
  return _comm_type;
}

// devices by dim for collective communication
DeviceGroup CommOpImpl::get_devices_by_dim(Operator& op, int32_t dim) const {
  Tensor& input = op->input(0);
  const auto& placement_group = src_group(op);
  const auto& placement = op->placement();
  HT_ASSERT(!placement_group.empty()) 
    << "Placement Group should be assigned before get devices by dim " << dim;
  HT_ASSERT(placement_group.contains(placement))
    << "Func get_devices_by_dim can only be called by device in src group: " 
    << placement_group << ", now get device " << placement << " in dst group!";

  int32_t local_device_idx = placement_group.get_index(placement);
  const auto& src_ds = input->get_distributed_states();
  const auto& order = src_ds.get_order();
  const auto& states = src_ds.get_states();

  auto idx = std::find(order.begin(), order.end(), dim);
  int32_t interval = 1;
  for (auto cur_order = idx + 1; cur_order != order.end(); cur_order++) {
    interval *= states.at(*cur_order);
  }
  int32_t macro_interval = interval * src_ds.get_dim(dim);
  int32_t start = local_device_idx - local_device_idx % macro_interval + local_device_idx % interval;
  std::vector<Device> comm_group;
  for (auto i = start; i < start + macro_interval; i += interval) {
    comm_group.push_back(placement_group.get(i));
  }
  return std::move(DeviceGroup(comm_group));
}

void CommOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta) const {
  const Tensor& input = inputs.at(0);
  Tensor& output = outputs.at(0);
  const auto& ds_input = input->get_distributed_states();
  const auto& ds_dst = get_dst_distributed_states(input->producer());
  // TODO: check states/order between src and dst
  HT_ASSERT(ds_input.is_valid() && ds_dst.is_valid())
          << "distributed states for input and dst tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_dst.get_device_num())
          << "cannot convert src distributed states to unpaired dst distributed states!";
  output->set_distributed_states(ds_dst);
}

bool CommOpImpl::DoMapToParallelDevices(Operator& op,
                                        const DeviceGroup& pg) const {
  const DeviceGroup& src_group = op->input(0)->placement_group();
  // set output statuses
  if (is_inter_group(op)) {
    std::vector<Device> devices;
    devices.insert(devices.end(), src_group.devices().begin(), src_group.devices().end());
    devices.insert(devices.end(), _dst_group.devices().begin(), _dst_group.devices().end());
    DeviceGroup merge_group(devices);
    op->instantiation_ctx().placement_group = merge_group;
    Operator::for_each_output_tensor(
      op, [&](Tensor& tensor) { tensor->set_placement_group(_dst_group); });
  } else {
    op->instantiation_ctx().placement_group = src_group;
    Operator::for_each_output_tensor(
      op, [&](Tensor& tensor) { tensor->set_placement_group(src_group); });    
  }
  return true;  
}

// todo: unuse comm ops should be removed before do intantiate
bool CommOpImpl::DoInstantiate(Operator& op, const Device& placement,
                               StreamIndex stream_index) const {
  const DistributedStates& src_ds = op->input(0)->get_distributed_states();
  const DistributedStates& dst_ds = get_dst_distributed_states(op);
  const DeviceGroup& src_group = op->input(0)->placement_group();
  // CommOp should be checked in Instantiate(when placement info assigned) whether it is valid  
  // HT_ASSERT(!src_ds.check_equal(dst_ds) || (!_dst_group.empty() && src_group != _dst_group))
  //   << "CommOp must communicate intra/inter device group!"
  //   << " src ds = " << src_ds.ds_info() << ", dst ds = " << dst_ds.ds_info()
  //   << ", src_group = " << src_group << ", dst_group = " << _dst_group;

  // create cuda events for used comm ops; if unused comm ops, will be substitute into None, just ignore here
  if (!src_ds.check_equal(dst_ds) || (!_dst_group.empty() && src_group != _dst_group)) {
    auto& inst_ctx = op->instantiation_ctx();
    inst_ctx.placement = placement;
    inst_ctx.stream_index = stream_index;
    if (placement.is_cuda()) {
      for (size_t i = 0; i < HT_MAX_NUM_MICRO_BATCHES; i++) { 
        inst_ctx.start[i] = std::make_unique<hetu::impl::CUDAEvent>(placement);
        inst_ctx.stop[i] = std::make_unique<hetu::impl::CUDAEvent>(placement);
      }
    } else {
      for (size_t i = 0; i < HT_MAX_NUM_MICRO_BATCHES; i++) {     
        inst_ctx.start[i] = std::make_unique<hetu::impl::CPUEvent>();
        inst_ctx.stop[i] = std::make_unique<hetu::impl::CPUEvent>();
      }
    }
  }
  Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
    if (tensor->placement_group().contains(placement)) {
      tensor->set_placement(placement);
    }
  });
  return true;
}

std::vector<NDArrayMeta> 
CommOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  const HTShape& input_shape = input->shape();
  const DistributedStates& src_ds = input->get_distributed_states();
  const DistributedStates& dst_ds = get_dst_distributed_states(input->producer());
  HTShape shape(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  return {NDArrayMeta().set_dtype(input->dtype()).set_device(input->device()).set_shape(shape)};
}


HTShapeList CommOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes,
                                     RuntimeContext& runtime_ctx) const {
  const HTShape& input_shape = input_shapes.at(0);
  Tensor& input = op->input(0);
  const auto& src_ds = input->get_distributed_states();
  const auto& dst_ds = get_dst_distributed_states(op);
  HTShape shape(input_shape.size());
  HT_LOG_DEBUG << "CommOpImpl::DoInferShape, src_ds = " << src_ds.get_states()
    << " and dst_ds = " << dst_ds.get_states();
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  HT_LOG_DEBUG << "CommOpImpl::DoInferShape, shape = " << shape;
  return {shape};
}

// todo: need to support multi ds
TensorList CommOpImpl::DoGradient(Operator& op,
                                  const TensorList& grad_outputs) const {
  // if input not requires grad, then grad_output also will be Tensor()                                    
  if (!op->requires_grad(0))
    return {Tensor()};                                    
  Tensor& input = op->input(0);
  Tensor& output = op->output(0);
  const Tensor& grad_output = grad_outputs.at(0);
  auto& graph = input->graph();
  DistributedStatesList multi_dst_ds;
  bool is_need_comm_op = false;
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
    graph.CUR_STRATEGY_ID = cur_strategy_id;
    const auto& ds_input = input->get_distributed_states();
    const auto& ds_output = output->get_distributed_states();
    const auto& ds_grad_output = grad_output->get_distributed_states();
    HT_ASSERT(ds_input.is_valid() && ds_output.is_valid())
            << "distributed states for input and output tensor must be valid!";
    HT_ASSERT(ds_input.get_device_num() == ds_output.get_device_num())
            << "distributed states for input and output tensor must be matched!";
    DistributedStates ds_grad_input(ds_input);
    if (ds_grad_input.get_dim(-2) > 1) { // partial->duplicate
      std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
      auto res_states = ds_grad_input.combine_states(src2dst);
      auto res_order = ds_grad_input.combine_order(src2dst);
      auto device_num = ds_grad_input.get_device_num();
      ds_grad_input.set_distributed_states({device_num, res_states, res_order});
    }
    multi_dst_ds.push_back(ds_grad_input);
    if (!ds_grad_input.check_equal(ds_grad_output)) {
      is_need_comm_op = true;
    }
  }
  graph.CUR_STRATEGY_ID = 0;
  // if forward just make partial into dup, then backward was dup to dup, needn't new comm_op
  if (!is_need_comm_op) {
    return {grad_output};
  } else {
    Tensor grad_input = MakeCommOp(grad_output, multi_dst_ds, OpMeta().set_name("grad_" + op->name()));
    return {grad_input};
  }
}

bool AllReduceOpImpl::DoMapToParallelDevices(Operator& op, 
                                             const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "AllReduceOp: device in comm_group: " << _comm_group.get(i) 
      << " must in palcement_group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);
}

bool AllReduceOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                    StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);
  auto ranks = DeviceGroupToWorldRanks(_comm_group);
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
AllReduceOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {inputs[0]->meta()};
}

HTShapeList AllReduceOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& runtime_ctx) const {
  return {input_shapes.at(0)};
}

NDArrayList AllReduceOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       RuntimeContext& ctx) const {
  // NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArrayList outputs = inputs; // just inplace here
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllReduce, inputs.at(0),
                                  outputs.at(0), reduction_type(), _comm_group, // _comm_group is a subset of placement_group
                                  op->instantiation_ctx().stream());
  return outputs;
}

// void AllReduceOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
//                                 NDArrayList& outputs, RuntimeContext& runtime_ctx) const {
//   HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
//                                   hetu::impl::AllReduce, inputs.at(0),
//                                   outputs.at(0), _comm_group, // _comm_group is a subset of placement_group
//                                   op->instantiation_ctx().stream());                              
// }

bool P2PSendOpImpl::DoMapToParallelDevices(Operator& op,
                                           const DeviceGroup& pg) const {
  HT_ASSERT(pg.num_devices() == _dst_group.num_devices())
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << pg << " vs. " << _dst_group;
  return OpInterface::DoMapToParallelDevices(op, pg);                                          
}


bool P2PSendOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                    StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);
  size_t dst_device_index = _dst_device_index == -1 ? 
         op->placement_group().get_index(op->placement()) : _dst_device_index;  
  auto src_rank = GetWorldRank();
  auto dst_rank = DeviceToWorldRank(_dst_group.get(dst_device_index));
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
P2PSendOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {};
}

HTShapeList P2PSendOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& runtime_ctx) const {
  return {};
}

void P2PSendOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& runtime_ctx) const {
  NDArray input = inputs.at(0);
  HT_ASSERT(input->dtype() == op->input(0)->dtype())
    << "Data type mismatched for P2P communication: " << input->dtype()
    << " vs. " << op->input(0)->dtype();
  size_t dst_device_index = _dst_device_index == -1 ? 
         op->placement_group().get_index(op->placement()) : _dst_device_index;

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), 
                                  type(), hetu::impl::P2PSend, input,
                                  _dst_group.get(dst_device_index), 
                                  op->instantiation_ctx().stream());                                 
}

bool P2PRecvOpImpl::DoMapToParallelDevices(Operator& op,
                                           const DeviceGroup& pg) const {
  HT_ASSERT(pg.num_devices() == _src_group.num_devices())
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << _src_group << " vs. " << pg;
  return OpInterface::DoMapToParallelDevices(op, pg);                                          
}

bool P2PRecvOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                  StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);
  size_t src_device_index = _src_device_index == -1 ?
         op->placement_group().get_index(op->placement()) : _src_device_index;
  auto src_rank = DeviceToWorldRank(_src_group.get(src_device_index));
  auto dst_rank = GetWorldRank();
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
P2PRecvOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {NDArrayMeta().set_dtype(_dtype).set_shape(_shape)};
}

HTShapeList P2PRecvOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& runtime_ctx) const {
  return {_shape};
}

void P2PRecvOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& runtime_ctx) const {
  size_t src_device_index = _src_device_index == -1 ?
         op->placement_group().get_index(op->placement()) : _src_device_index;

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::P2PRecv, outputs.at(0),
                                  _src_group.get(src_device_index),
                                  op->instantiation_ctx().stream());
}

bool BatchedISendIRecvOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                            StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);                                      
  std::vector<int> ranks(_comm_devices.size());
  std::transform(_comm_devices.begin(), _comm_devices.end(), ranks.begin(), [&](const Device& device) { return DeviceToWorldRank(device); });
  std::sort(ranks.begin(), ranks.end());
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
BatchedISendIRecvOpImpl::DoInferMeta(const TensorList& inputs) const {
  if (_outputs_shape.size() == 0)
    return {};
  std::vector<NDArrayMeta> output_meta_lsit;
  for (auto& output_shape: _outputs_shape) {
    output_meta_lsit.push_back(NDArrayMeta().set_dtype(_dtype).set_shape(output_shape));
  }
  return std::move(output_meta_lsit);
}

HTShapeList BatchedISendIRecvOpImpl::DoInferShape(Operator& op, 
                                                  const HTShapeList& input_shapes,
                                                  RuntimeContext& runtime_ctx) const {
  if (_outputs_shape.size() == 0)
    return {};                                                    
  HTShapeList outputs_shape(_outputs_shape);                                                    
  return std::move(outputs_shape);
}  

// deprecated: only used in gpt inference, before symbolic shape is realized
HTShapeList BatchedISendIRecvOpImpl::DoInferDynamicShape(Operator& op, 
                                                  const HTShapeList& input_shapes,
                                                  RuntimeContext& runtime_ctx) const {                                             
  HTShapeList outputs_shape(input_shapes);                                                    
  return std::move(outputs_shape);
}  

void BatchedISendIRecvOpImpl::DoCompute(Operator& op, 
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs, 
                                        RuntimeContext& runtime_ctx) const {
  for (int i = 0; i < op->num_inputs(); i++) {
    const NDArray& input = inputs.at(i);
    HT_ASSERT(input->dtype() == op->input(i)->dtype())
      << "Data type mismatched for ISend communication: " << input->dtype()
      << " vs. " << op->input(i)->dtype();
  }
  // NOTE: For communication ops, we insert Contiguous op during `MakeOp()`
  // to ensure inputs are contiguous. But for BatchedISendIRecv, we found
  // that inputs may be non-contiguous, which is weird. So we make them
  // contiguous again here.
  NDArrayList contig_inputs;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(contig_inputs), [&](const NDArray& input) {
    return input->is_contiguous() ? input : NDArray::contiguous(input, op->instantiation_ctx().stream_index);
  });

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), 
                                  hetu::impl::BatchedISendIRecv, contig_inputs, _dst_devices, outputs, 
                                  _src_devices, _comm_devices, op->instantiation_ctx().stream());
}

bool AllGatherOpImpl::DoMapToParallelDevices(Operator& op,
                                             const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "Allgather: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);  
}

bool AllGatherOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                    StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);                                      
  auto ranks = DeviceGroupToWorldRanks(_comm_group);
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
AllGatherOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  DataType dtype = input->dtype();
  HTShape gather_shape = input->shape();
  gather_shape[0] *= _comm_group.num_devices();
  return {NDArrayMeta().set_dtype(dtype).set_shape(gather_shape)};
}

HTShapeList AllGatherOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& runtime_ctx) const {
  HTShape gather_shape = input_shapes.at(0);
  gather_shape[0] *= _comm_group.num_devices();
  return {gather_shape};  
}

void AllGatherOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& runtime_ctx) const {
  HT_ASSERT(inputs.at(0)->dtype() == op->input(0)->dtype())
    << "Data type mismatched for AllGather communication: " << inputs.at(0)->dtype()
    << " vs. " << op->input(0)->dtype();

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllGather, inputs.at(0), outputs.at(0), 
                                  _comm_group, op->instantiation_ctx().stream());
}

bool ReduceScatterOpImpl::DoMapToParallelDevices(Operator& op,
                                                 const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "ReduceScatter: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);  
}

bool ReduceScatterOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                        StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);                                      
  auto ranks = DeviceGroupToWorldRanks(_comm_group);
  NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream());
  return ret;
}

std::vector<NDArrayMeta> 
ReduceScatterOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  DataType dtype = input->dtype();
  HTShape scatter_shape = input->shape();
  scatter_shape[0] /= _comm_group.num_devices();
  HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
    << input->shape()[0] << " must >= comm devices num: " << _comm_group.num_devices();  
  return {NDArrayMeta().set_dtype(dtype).set_shape(scatter_shape)};
}

HTShapeList ReduceScatterOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes,
                                              RuntimeContext& runtime_ctx) const {
  HTShape scatter_shape = input_shapes.at(0);
  scatter_shape[0] /= _comm_group.num_devices();
  HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
    << input_shapes.at(0)[0] << " must >= comm devices num: " << _comm_group.num_devices();  
  return {scatter_shape};
}

NDArrayList ReduceScatterOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& ctx) const {                              
  NDArrayList outputs = {};
  HT_ASSERT(inputs.at(0)->dtype() == op->input(0)->dtype())
    << "Data type mismatched for ReduceScatter communication: " << inputs.at(0)->dtype()
    << " vs. " << op->input(0)->dtype();

  // if (inplace()) {
  // just inplace here
  NDArrayMeta meta = inputs.at(0)->meta();
  HTShape scatter_shape = inputs.at(0)->shape();
  scatter_shape[0] /= _comm_group.num_devices();
  meta.set_shape(scatter_shape);
  // int rank = GetWorldRank();
  int rank = _comm_group.get_index(op->placement());
  size_t storage_offset = rank * (inputs.at(0)->numel() / _comm_group.num_devices());
  NDArray output = NDArray(meta, inputs.at(0)->storage(), inputs.at(0)->storage_offset() + storage_offset);
  outputs.emplace_back(output);
  // }
  // else {
  //   outputs = DoAllocOutputs(op, inputs, ctx);
  // }
  
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReduceScatter, inputs.at(0), outputs.at(0), 
                                  reduction_type(), _comm_group, op->instantiation_ctx().stream());
  return outputs;
}

// void ReduceScatterOpImpl::DoCompute(Operator& op, 
//                                     const NDArrayList& inputs,
//                                     NDArrayList& outputs,
//                                     RuntimeContext& runtime_ctx) const {
//   HT_ASSERT(inputs.at(0)->dtype() == op->input(0)->dtype())
//     << "Data type mismatched for ReduceScatter communication: " << inputs.at(0)->dtype()
//     << " vs. " << op->input(0)->dtype();

//   HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
//                                   hetu::impl::ReduceScatter, inputs.at(0), outputs.at(0), 
//                                   _comm_group, op->instantiation_ctx().stream());
// }

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, 
                  ReductionType red_type, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<CommOpImpl>(std::move(multi_dst_ds), DeviceGroup(), red_type), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds,
                  const std::string& mode, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<CommOpImpl>(std::move(multi_dst_ds), DeviceGroup(), Str2ReductionType(mode)), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, DeviceGroup dst_group, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<CommOpImpl>(std::move(multi_dst_ds), std::move(dst_group)), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<CommOpImpl>(std::move(multi_dst_ds)), 
                      {input}, std::move(op_meta))->output(0);
}

// for comm ops created in exec_graph, the device_groups only contains one device_group!!!
Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, bool inplace, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].is_subset(comm_group))) << "comm_group must be subset of device_group!";
  return Graph::MakeOp(std::make_shared<AllReduceOpImpl>(std::move(comm_group), kSUM, inplace), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, 
                       ReductionType red_type, bool inplace, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].is_subset(comm_group))) << "comm_group must be subset of device_group!";
  return Graph::MakeOp(std::make_shared<AllReduceOpImpl>(std::move(comm_group), red_type, inplace), 
                      {input}, std::move(op_meta))->output(0);
}

// p2p send no output
Tensor MakeP2PSendOp(Tensor input, DeviceGroup dst_group, 
                     int dst_device_index, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].num_devices() == dst_group.num_devices()))
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << op_meta.device_groups[0] << " vs. " << dst_group;
  return Graph::MakeOp(std::make_shared<P2PSendOpImpl>(std::move(dst_group), 
                       dst_device_index), {input}, std::move(op_meta))->out_dep_linker();
}

Tensor MakeP2PRecvOp(DeviceGroup src_group, DataType dtype,
                     HTShape shape, int src_device_index, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].num_devices() ==src_group.num_devices()))
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << op_meta.device_groups[0] << " vs. " << src_group;
  return Graph::MakeOp(std::make_shared<P2PRecvOpImpl>(std::move(src_group), dtype, std::move(shape), 
                       src_device_index), {}, std::move(op_meta))->output(0);
}

/*
// symbolic shape
Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               const std::vector<Device>& dst_devices, 
                               const SyShapeList& outputs_shape, 
                               const std::vector<Device>& src_devices, 
                               const std::vector<Device>& comm_devices, 
                               DataType dtype, OpMeta op_meta) {
  if (src_devices.size() == 0)
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(dst_devices, outputs_shape,
                        src_devices, comm_devices, dtype), std::move(inputs), std::move(op_meta))->out_dep_linker();
  else
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(dst_devices, outputs_shape,
                        src_devices, comm_devices, dtype), inputs, std::move(op_meta))->output(0);  
}
*/

// fixed shape
Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               std::vector<Device> dst_devices, 
                               HTShapeList outputs_shape, 
                               std::vector<Device> src_devices, 
                               std::vector<Device> comm_devices, 
                               DataType dtype, OpMeta op_meta) {
  if (src_devices.size() == 0)
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(std::move(dst_devices), std::move(outputs_shape),
                        std::move(src_devices), std::move(comm_devices), dtype), std::move(inputs), std::move(op_meta))->out_dep_linker();
  else
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(std::move(dst_devices), std::move(outputs_shape),
                        std::move(src_devices), std::move(comm_devices), dtype), std::move(inputs), std::move(op_meta))->output(0);  
}

Tensor MakeAllGatherOp(Tensor input, DeviceGroup comm_group, 
                       OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].is_subset(comm_group))) << "comm_group must be subset of device_group!";
  return Graph::MakeOp(std::make_shared<AllGatherOpImpl>(std::move(comm_group)), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeReduceScatterOp(Tensor input, DeviceGroup comm_group, 
                           bool inplace, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].is_subset(comm_group))) << "comm_group must be subset of device_group!";
  return Graph::MakeOp(std::make_shared<ReduceScatterOpImpl>(std::move(comm_group), kSUM, inplace), 
                      {input}, std::move(op_meta))->output(0);
}

Tensor MakeReduceScatterOp(Tensor input, DeviceGroup comm_group, 
                           ReductionType red_type, bool inplace, OpMeta op_meta) {
  HT_ASSERT(op_meta.device_groups.empty() || (op_meta.device_groups.size() == 1 
    && op_meta.device_groups[0].is_subset(comm_group))) << "comm_group must be subset of device_group!";
  return Graph::MakeOp(std::make_shared<ReduceScatterOpImpl>(std::move(comm_group), red_type, inplace), 
                      {input}, std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
