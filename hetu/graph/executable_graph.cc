#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

Operator& ExecutableGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                       TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  return MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
}

void ExecutableGraph::ResetVariableDataInner(const Tensor& tensor,
                                             const Initializer& init) {
  if (tensor->placement().is_undetermined()) {
    _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  } else {
    init.Init(GetVariableDataInner(tensor));
  }
}

NDArray& ExecutableGraph::GetVariableDataInner(const Tensor& tensor) {
  auto it = _preserved_data.find(tensor->id());
  HT_RUNTIME_ERROR_IF(it == _preserved_data.end())
    << "Cannot find data for variable tensor " << tensor;
  return it->second;
}

NDArray& ExecutableGraph::AllocVariableDataInner(const Tensor& tensor,
                                                 const Initializer& init) {
  // TODO: check meta is valid
  _preserved_data[tensor->id()] =
    NDArray::empty(tensor->shape(), tensor->placement(), tensor->dtype());
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()]);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()]);
  }
  return _preserved_data[tensor->id()];
}

void ExecutableGraph::RegisterVariableDataInner(const Tensor& tensor,
                                                NDArray data,
                                                const Initializer& init) {
  _preserved_data[tensor->id()] = std::move(data);
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()]);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()]);
  }
}

// bool ExecutableGraph::MapOpsToParallelDevices(
//   const DeviceGroup& placement_group) {
//   HT_NOT_IMPLEMENTED;
//   return true;
// }

bool ExecutableGraph::Instantiate(const TensorList& fetches,
                                  const Device& preferred_device) {
  auto is_op_instantiated = [&](const Operator& op) -> bool {
    return !op->placement().is_undetermined();
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);
  HT_LOG_TRACE << "Instantiating ops: " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (!op->placement().is_undetermined())
      continue;
    
    // 1. for global info: op->placement_group + tensor->placement_group
    if (!op->device_group().empty()) {
      op->MapToParallelDevices(op->device_group());
    } else {
      DeviceGroup inferred;
      if (is_group_op(op)) {
        std::vector<Device> devices;
        for (auto& input : op->in_dep_linkers()) {
          for (auto& device : input->producer()->placement_group().devices()) {
            devices.push_back(device);
          }
        }
        inferred = DeviceGroup(devices);
      } else {
        HT_ASSERT(op->num_inputs() > 0)
          << "Currently we cannot infer the devices "
          << "for operators with zero in-degree. : " << op;
        inferred = op->input(0)->producer()->placement_group();        
      }
      op->MapToParallelDevices(inferred);
    }

    // 2. for local compute: op->placement + tensor->placement
    if (!op->placement_group().contains(preferred_device))
      continue; // pipeline parallel
    Device placement =
      is_device_to_host_op(op) ? Device(kCPU) : preferred_device;
    StreamIndex stream_id = get_suggested_stream_index(op);
    HT_LOG_TRACE << "Instantiating op " << op << " (placement=" << placement
                 << ", stream_index=" << stream_id << ")";
    bool ok = op->Instantiate(placement, stream_id);
    if (!ok && !placement.is_cpu()) {
      HT_LOG_WARN << "Failed to instantiate op " << op << " on " << placement
                  << ". Will try to instantiate it on the host device.";
      placement = Device(kCPU);
      ok = op->Instantiate(placement, stream_id);
    }
    HT_VALUE_ERROR_IF(!ok) << "Failed to instantiate op " << op << " on "
                           << placement;

    // remove duplicate comm ops
    if (is_comm_op(op)) {
      auto& input_op = op->input(0)->producer();
      if (is_comm_op(input_op)) {
        ReplaceInput(op, 0, input_op->input(0));
        reinterpret_cast<CommOpImpl&>(op->body()).get_comm_type(op); // input changes, update comm_op type
      }
    }
    
    // add transfer ops
    for (size_t i = 0; i < op->num_inputs(); i++) {
      auto& input = op->input(i);
      if (input->placement() != placement) {
        HT_RUNTIME_ERROR_IF(!input->placement().local())
          << "Please use P2P communication to fetch remote input";

        auto& input_op = input->producer();

        Tensor transferred_input;
        StreamIndex transfer_stream_id;
        if (input->placement().is_cpu()) {
          transferred_input = MakeDataH2DOp(placement, input);
          transfer_stream_id = kH2DStream;
        } else if (placement.is_cpu()) {
          transferred_input = MakeDataD2HOp(placement, input);
          transfer_stream_id = kD2HStream;
        } else {
          // TODO: support cuda memcpy across processes
          HT_NOT_IMPLEMENTED << "We should use NCCL for P2P communication.";
          __builtin_unreachable();
        }
        auto& transfer_op = transferred_input->producer();
        if (!input_op->placement_group().empty())
          transfer_op->MapToParallelDevices(input_op->placement_group());
        transfer_op->Instantiate(placement, transfer_stream_id);
        ReplaceInput(op, i, transferred_input);
      }
    }
  }

  return true;
}

void ExecutableGraph::SubstituteCommOp(const OpRefList& topo_order) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    if (is_comm_op(op)) {
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op begin: " << op << "...";
      auto& comm_op = op;
      auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(op->body());
      uint64_t comm_type = comm_op_impl.get_comm_type(op);
      Tensor result;
      if (comm_type == COMM_SPLIT_OP) {
        auto local_device_index =comm_op->placement_group().get_index(comm_op->placement());
        const auto& dst_ds = comm_op_impl.get_dst_distributed_states();
        auto cur_state_index = dst_ds.map_device_to_state_index(local_device_index);
        const auto& order = dst_ds.get_order();
        const auto& states = dst_ds.get_states();
        HTAxes keys; 
        HTShape indices, splits;
        for (auto o : order) {
          if (o >= 0) { 
            keys.push_back(o);
            splits.push_back(states.at(o));
            indices.push_back(cur_state_index[o]);
          }
        }
        HT_LOG_DEBUG << local_device << ": keys = " << keys << "; indices = " << indices << "; splits = " << splits;
        Tensor split_output = MakeSplitOp(comm_op->input(0), keys, indices, splits, OpMeta().set_is_deduce_states(false));
        auto& split_op = split_output->producer();
        split_op->MapToParallelDevices(comm_op->placement_group());
        split_op->Instantiate(local_device, kComputingStream);
        result = split_output;
      } else if (comm_type == ALL_REDUCE_OP) {
        DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2); // do allreduce among comm_group
        Tensor all_reduce_output = MakeAllReduceOp(
          comm_op->input(0), comm_group, // comm_group is a subset of comm_op's placement_group
          OpMeta().set_device_group(comm_op->placement_group())
                  .set_is_deduce_states(false)
                  .set_name(comm_op->input(0)->name() + "_AllReduce"));
        auto& all_reduce_op = all_reduce_output->producer();
        all_reduce_op->MapToParallelDevices(comm_op->placement_group());
        all_reduce_op->Instantiate(local_device, kCollectiveStream);
        result = all_reduce_output;
        HT_LOG_DEBUG << local_device << ": substitute comm_op to all_reduce_op: " << comm_group;        
      } else if (comm_type == ALL_GATHER_OP) {
        DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, 0);
        Tensor all_gather_output = MakeAllGatherOp(
          comm_op->input(0), comm_group, comm_op->device_group(),
          OpMeta().set_device_group(comm_op->placement_group())
                  .set_is_deduce_states(false)
                  .set_name(comm_op->input(0)->name() + "_AllGather"));
        auto& all_gather_op = all_gather_output->producer();
        all_gather_op->MapToParallelDevices(comm_op->placement_group());
        all_gather_op->Instantiate(local_device, kCollectiveStream);
        result = all_gather_output;
        HT_LOG_DEBUG << local_device << ": substitute comm_op to all_gather_op: " << comm_group;
      } else if (comm_type == REDUCE_SCATTER_OP) {
        DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2);
        Tensor reduce_scatter_output =  MakeReduceScatterOp(
          comm_op->input(0), comm_group, comm_op->device_group(),
          OpMeta().set_device_group(comm_op->placement_group())
                  .set_is_deduce_states(false)
                  .set_name(comm_op->input(0)->name() + "_ReduceScatter"));
        auto& reduce_scatter_op = reduce_scatter_output->producer();
        reduce_scatter_op->MapToParallelDevices(comm_op->placement_group());
        reduce_scatter_op->Instantiate(local_device, kCollectiveStream);
        result = reduce_scatter_output;
        HT_LOG_DEBUG << local_device << ": substitute comm_op to reduce_scatter_op: " << comm_group;
      } else if (comm_type == P2P_OP) {
        // 1. local_device send data to other devices 2. local_device recv data from other devices
        DataType dtype = comm_op->input(0)->dtype();
        int32_t local_device_index = comm_op->placement_group().get_index(comm_op->placement());
        TensorList send_datas_local;
        std::vector<int32_t> dsts_local;
        HTShapeList recv_shapes_local;
        std::vector<int32_t> srcs_local;
        Tensor self_send_data;
        std::vector<std::pair<int32_t, int32_t>> send_pairs;
        for (int32_t used_device_index = 0; used_device_index < comm_op->placement_group().num_devices(); used_device_index++) {     
          HT_LOG_DEBUG << local_device << ": cross send begin!";
          int32_t device_index = 0;
          TensorList send_datas;
          std::vector<int32_t> dsts;
          // execute cross_send for all devices to get the complete recv_shapes
          CrossSend({}, {}, 0, false, device_index, comm_op, send_datas, dsts, used_device_index);
          HT_ASSERT(device_index == comm_op->placement_group().num_devices()) << "cross send error!";
          HT_LOG_DEBUG << local_device << ": cross send end!";
          // for batch send/recv
          for (int i = 0; i < dsts.size(); i++) {
            send_pairs.push_back({used_device_index, dsts[i]}); // for comm_set
            // local_device send to other devices
            if (used_device_index == local_device_index && dsts[i] != local_device_index) {
              send_datas_local.push_back(send_datas[i]);
              dsts_local.push_back(dsts[i]);
            } 
            // local device recv from other devices
            if (used_device_index != local_device_index && dsts[i] == local_device_index) {
              recv_shapes_local.push_back(send_datas[i]->shape());
              srcs_local.push_back(used_device_index);              
            }
            // special case: local device send to self
            if (used_device_index == local_device_index && dsts[i] == local_device_index) {
              self_send_data = send_datas[i];
            }
          }
        }

        // get comm_devices for batch isend/irecv, union set
        std::set<int32_t> comm_set;
        comm_set.insert(local_device_index);
        comm_set.insert(dsts_local.begin(), dsts_local.end());
        comm_set.insert(srcs_local.begin(), srcs_local.end());
        bool keep_search = true;
        while (keep_search) {
          keep_search = false;
          for (auto& pair : send_pairs) {
            bool find_first = (comm_set.find(pair.first) != comm_set.end());
            bool find_second = (comm_set.find(pair.second) != comm_set.end());
            if (find_first && !find_second) {
              comm_set.insert(pair.second);
              keep_search = true;
            } else if (!find_first && find_second) {
              comm_set.insert(pair.first);
              keep_search = true;
            }
          }
        }
        std::vector<Device> comm_devices(comm_set.size());
        std::vector<Device> dst_devices(dsts_local.size());
        std::vector<Device> src_devices(srcs_local.size());
        std::transform(dsts_local.begin(), dsts_local.end(), dst_devices.begin(), [&](int32_t index) { return comm_op->placement_group().get(index); });
        std::transform(srcs_local.begin(), srcs_local.end(), src_devices.begin(), [&](int32_t index) { return comm_op->placement_group().get(index); });        
        std::transform(comm_set.begin(), comm_set.end(), comm_devices.begin(), [&](int32_t index) { return comm_op->placement_group().get(index); });
        // HT_LOG_DEBUG << local_device << ": MakeBatchedISendIRecvOp for " << comm_op->name() 
        //                              << ": send_datas_local=" << send_datas_local
        //                              << ", dst_devices=" << DeviceGroup(dst_devices)
        //                              << ", recv_shapes=" << recv_shapes_local
        //                              << ", src_devices=" << DeviceGroup(src_devices) 
        //                              << ", comm_devices=" << DeviceGroup(comm_devices);
        // TODO: when needn't recv, MakeBatchedISendIRecvOp cannot return null output tensor, how to get the MakeBatchedISendIRecvOp???                                       
        Tensor batched_isend_irecv_output = MakeBatchedISendIRecvOp(send_datas_local, dst_devices, recv_shapes_local, src_devices, comm_devices, dtype, 
          OpMeta().set_is_deduce_states(false).set_name("BatchedISendIRecvOp_for_" + comm_op->name()));
        auto& batched_isend_irecv_op = batched_isend_irecv_output->producer();
        batched_isend_irecv_op->MapToParallelDevices(comm_op->placement_group());
        batched_isend_irecv_op->Instantiate(local_device, kCollectiveStream);
        if (dst_devices.size() == 0) { // connect comm_op->input producer with batchISendIRecvOp when needn't send
          AddInDeps(batched_isend_irecv_op, {comm_op->input(0)});
          HT_LOG_DEBUG << local_device << ": BatchISendIRecv needn't send, so connect "
                       << batched_isend_irecv_op << "with comm_op's input " << comm_op->input(0);
        }
        if (src_devices.size() == 0) { // connect batchISendIRecvOp with comm_op->ouput consumers when needn't recv
          for (int i = 0; i < comm_op->output(0)->num_consumers(); i++) {
            AddInDeps(comm_op->output(0)->consumer(i), {batched_isend_irecv_op->out_dep_linker()});
          }
        }
        TensorList recv_datas_local = batched_isend_irecv_op->outputs();

        HT_LOG_DEBUG << local_device << ": cross receive begin!";
        int32_t device_index = 0;
        // already get the recv_datas by batch_send_recv, so just need local device to execute cross_receive
        result = CrossReceive(0, device_index, comm_op, recv_datas_local, srcs_local, self_send_data, local_device_index);
        HT_ASSERT(device_index == comm_op->placement_group().num_devices()) << "cross receive error!";
        HT_LOG_DEBUG << local_device << ": cross receive end!";    
      }
      result->set_distributed_states(comm_op_impl.get_dst_distributed_states()); // assign distributed states for result tensor

      // find all comm_op->output consumers, and replace the correspond input tensor with result tensor
      for (int i = comm_op->output(0)->num_consumers() - 1; i >= 0; i--) {
        auto& consumer_i = comm_op->output(0)->consumer(i);
        for (int j = 0; j < consumer_i->num_inputs(); j++) {
          if (consumer_i->input(j)->id() == comm_op->output(0)->id()) {
            ReplaceInput(consumer_i, j, result);
          }
        }
      }
      // comm_op->output(0) = result;
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op end: " << op << "...";
    }
  }
}

Tensor ExecutableGraph::CrossReceive(int32_t depth, int32_t& device_index, Operator& comm_op, 
                                     TensorList& recv_datas, std::vector<int32_t>& srcs,
                                     Tensor& self_send_data, int32_t& used_device_index) {
  HT_ASSERT(is_comm_op(comm_op)) << comm_op->name() << " must be comm_op!";
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());
  const auto& prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  const auto& prev_order = prev_distributed_states.get_order();
  auto loop_sizes = prev_distributed_states.get_loop_sizes();

  const auto& target_distributed_states = comm_op_impl.get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  HT_ASSERT(comm_op->placement() == local_device) 
          << "Corss Receive: comm_op's placement " << comm_op->placement() 
          << " != " << "local device " << local_device;
  auto local_device_index = comm_op->placement_group().get_index(comm_op->placement());
  auto cur_state_index = target_distributed_states.map_device_to_state_index(used_device_index); // 指定的device需要的是tensor的哪一部分数据

  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };
  
  Tensor result;
  // cur_state_index存的是used device需要的是哪些数据, 最终的result是从device_index对应的device中concatenate/sum获取而来的
  if (depth == prev_order.size()) {
    // 如果recv的对象就是自己, 则无需send/recv op
    if (device_index == used_device_index) {
      // 判断self_send_data是否已经赋值
      HT_ASSERT(self_send_data.is_defined()) << "Cross Receive: self_send_data must be a valid tensor!";
      result = self_send_data;
      HT_LOG_DEBUG << local_device << ": device " << used_device_index 
                   << ": recv from device " << device_index << " don't need irecv";
    } else {
      for (int i = 0; i < srcs.size(); i++) {
        if (srcs[i] == device_index) {
          result = recv_datas[i];
          break;
        }
      }
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": recv from device " << device_index;      
    }
    device_index += 1;            
  } else {
    auto cur_dim = prev_order[depth];
    if (cur_dim == -2) { // partial
      TensorList part_result_list;
      for (size_t i = 0; i < prev_partial; i++) {
        auto part_result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        part_result_list.push_back(part_result);
      }
      auto sum_output = MakeSumOp(part_result_list, OpMeta().set_is_deduce_states(false));
      auto& sum_op = sum_output->producer();
      if (used_device_index == local_device_index) {
        sum_op->MapToParallelDevices(comm_op->placement_group());
        sum_op->Instantiate(local_device, kComputingStream);
      }
      result = sum_output;    
    } else if (cur_dim == -1) {
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += ((target_duplicate - cur_st) * multiple - 1) * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += (target_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_states().at(cur_dim);
      auto tar_st = target_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        if (multiple == 1) {
          result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        } else {
          TensorList part_result_list;
          for (size_t i = 0; i < multiple; i++) {
            auto part_result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
            part_result_list.push_back(part_result);
          }
          auto concatenate_output = MakeConcatenateOp(part_result_list, cur_dim, OpMeta().set_is_deduce_states(false));
          auto& concatenate_op = concatenate_output->producer();
          if (used_device_index == local_device_index) {
            concatenate_op->MapToParallelDevices(comm_op->placement_group());
            concatenate_op->Instantiate(local_device, kComputingStream);
          }
          result = concatenate_output;
        }
        device_index += (tar_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += (tar_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    }
  }
  
  return result;  
}

void ExecutableGraph::CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                                std::unordered_map<int32_t, int32_t> split_target_state,
                                int32_t depth, bool need_split, int32_t& device_index, 
                                Operator& comm_op, TensorList& send_datas, 
                                std::vector<int32_t>& dsts, int32_t& used_device_index) {
  // basic info
  HT_ASSERT(is_comm_op(comm_op)) << comm_op->name() << " must be comm_op!";
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());  
  const auto& prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  HT_ASSERT(comm_op->placement() == local_device) 
          << "Corss Send: comm_op's placement " << comm_op->placement() 
          << " != " << "local device " << local_device;
  auto local_device_index =comm_op->placement_group().get_index(comm_op->placement());  
  auto cur_state_index = prev_distributed_states.map_device_to_state_index(used_device_index); // 根据指定的device index和order确定该device拥有的是tensor的哪部分数据

  const auto& target_distributed_states = comm_op_impl.get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  const auto& target_order = target_distributed_states.get_order();
  auto loop_sizes = target_distributed_states.get_loop_sizes();                  
  
  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };

  auto get_keys = [](std::unordered_map<int32_t, int32_t> map) -> HTAxes {
    HTAxes keys; 
    keys.reserve(map.size());
    for (auto kv : map) {
      keys.push_back(kv.first);
    }
    return keys;
  };

  // cross send part
  if (prev_partial == 1 && prev_duplicate > target_duplicate && get_state_index(-1) % (prev_duplicate / target_duplicate) != 0) {    
    HT_LOG_DEBUG << local_device << ": device " << used_device_index << " don't need to send to other devices!";
    device_index += comm_op->placement_group().num_devices();
    return;
  }  
  if (depth == target_order.size()) {
    Tensor send_part;
    if (need_split) {
      HTAxes keys = get_keys(split_target_state);
      HTShape indices, splits;
      indices.reserve(keys.size()); splits.reserve(keys.size());
      for (auto key : keys) {
        indices.push_back(split_cur_state[key]);
        splits.push_back(split_target_state[key]);
      }
      // split_op: 把tensor在keys这些dimension上按照splits[key]份数切分, 并取出第indices[key]份, 作为要send的数据切片
      auto split_output = MakeSplitOp(comm_op->input(0), keys, indices, splits, OpMeta().set_is_deduce_states(false));
      auto& split_op = split_output->producer();
      if (used_device_index == local_device_index) { // 其他device上生成的用于替换comm_op不需要map placement_group和placement
        split_op->MapToParallelDevices(comm_op->placement_group());
        split_op->Instantiate(local_device, kComputingStream);
      }
      send_part = split_output;
    } else {
      // 如果不需要split, 则发送整个tensor
      send_part = comm_op->input(0);
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index << " don't need split";      
    }
    if (device_index == used_device_index) {
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index << " don't need isend";
    } else {
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index;
    }    
    send_datas.push_back(send_part);
    dsts.push_back(device_index);

    device_index += 1;
  } else {
    auto cur_dim = target_order[depth];
    if (cur_dim < 0) {
      HT_ASSERT(cur_dim == -1) << "Target distributed states must not enable partial!";
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        if (cur_st % multiple != 0) {
          HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": don't need to send to other devices!";
          return;
        }
        device_index += cur_st / multiple * loop_sizes[depth];
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, comm_op, send_datas, dsts, used_device_index);
        device_index += (prev_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, comm_op, send_datas, dsts, used_device_index);
        }
        device_index += (prev_duplicate - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      auto tar_st = target_distributed_states.get_states().at(cur_dim);
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        split_cur_state[cur_dim] = 0;
        split_target_state[cur_dim] = 1;
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, comm_op, send_datas, dsts, used_device_index);
        device_index += (pre_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          split_cur_state[cur_dim] = i;
          split_target_state[cur_dim] = multiple; 
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, comm_op, send_datas, dsts, used_device_index);
        }
        device_index += (pre_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    }
  }
}

NDArrayList ExecutableGraph::Run(const TensorList& fetches,
                                 const FeedDict& feed_dict) {                        
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  HT_LOG_DEBUG << local_device << ": exec graph run begin .............";                              
  // TODO: For each pair of `fetches` and `feed_dict`,
  // deduce the optimal execution plan, and cache it.
  for (auto& fetch : fetches) {
    if (fetch->placement().is_undetermined()) {
      Instantiate(fetches, local_device);
      break;
    }
  }

  // init topo contains comm_op
  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_computed);
  HT_LOG_DEBUG << local_device << ": global topo before substitute comm_op: " << topo;

  // substitute comm_op
  HT_LOG_DEBUG << local_device << ": substitute comm_op begin...";
  Graph::push_graph_ctx(id()); // ensure the new ops created in execute_graph
  SubstituteCommOp(topo);
  Graph::pop_graph_ctx();
  HT_LOG_DEBUG << local_device << ": substitute comm_op end...";

  // update topo
  OpRefList updated_topo = Graph::TopoSort(fetches, -1, is_op_computed);
  HT_LOG_DEBUG << local_device << ": updated global topo after substitute comm_op: " << updated_topo;

  OpRefList local_topo;
  std::copy_if(updated_topo.begin(), updated_topo.end(), std::back_inserter(local_topo), 
  [&](OpRef& op_ref) { return op_ref.get()->placement_group().contains(local_device); });  
  HT_LOG_DEBUG << local_device << ": local topo after substitute comm_op: " << local_topo;

  // compute
  RuntimeContext runtime_ctx(local_topo.size());
  Tensor2NDArrayMap tensor2data;
  tensor2data.reserve(local_topo.size());
  tensor2data.insert(feed_dict.begin(), feed_dict.end());
  NDArrayList results(fetches.size());
  std::unordered_map<TensorId, size_t> fetch_indices;
  for (size_t i = 0; i < fetches.size(); i++)
    fetch_indices[fetches.at(i)->id()] = i;
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());

  for (auto& op_ref : local_topo) {
    auto& op = op_ref.get();
    // Question: Is it possible that some outputs are fed in
    // while the rest are not?
    bool computed = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
    if (computed)
      continue;

    HT_LOG_DEBUG << local_device << ": " << op->name() << " comupte begin...";

    NDArrayList inputs;
    inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++) {
      // TODO: Support async transfer. And this could be checked for once.
      auto& data = tensor2data[op->input(i)->id()];
      if (data->device() != op->input(i)->placement() ||
          data->dtype() != op->input(i)->dtype()) {
        tensor2data[op->input(i)->id()] =
          NDArray::to(data, op->input(i)->placement(), op->input(i)->dtype(),
                      kBlockingStream);
      }
      inputs.push_back(tensor2data[op->input(i)->id()]);
    }
    auto outputs = op->Compute(inputs, runtime_ctx);
    
    // op->Sync();
    // HT_LOG_DEBUG << local_device << ": " << op->name() << " comupte end..." 
    //   << "\ninputs = " << inputs 
    //   << "\noutputs = " << outputs;  

    for (size_t i = 0; i < outputs.size(); i++) {
      tensor2data.insert({op->output(i)->id(), outputs[i]});
    }
    
    Operator::for_each_output_tensor(op, [&](const Tensor& output) {
      auto it = fetch_indices.find(output->id());
      if (it != fetch_indices.end()) {
        if (output->output_id() >= 0)
          results[it->second] = outputs[output->output_id()];
        to_sync_op_ids.insert(op->id());
      }
    });
    // TODO: remove inputs that are no longer used
  }
  for (auto op_id : to_sync_op_ids) {
    _op_indexing[op_id]->Sync();
  }
  return results;
}

} // namespace graph
} // namespace hetu
