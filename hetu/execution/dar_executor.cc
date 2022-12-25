#include "hetu/execution/dar_executor.h"
#include "hetu/execution/device_placer.h"
#include "hetu/execution/run_metadata.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/dataloader.h"
#include "hetu/impl/communication/comm_group.h"
#include <queue>

namespace hetu {
namespace execution {

using hetu::operator<<;

DARExecutor::DARExecutor(const Device& local_device,
                         const DeviceGroup& device_group,
                         const TensorList& losses) {
  _exec_ctx = std::make_shared<DARExecutionContext>();
  _exec_ctx->_local_device = local_device;
  _exec_ctx->_device_group = device_group;
  // TODO: support garbage collection in OperatorPool
  // 0. 所有tensor的placement_group/placement/index和device_num的赋值在这个阶段完成
  bool parallel = PlaceDevices(TopoSort(OperatorPool::GetAllOps(), true));
  _exec_ctx->_global_topo_order = TopoSort(OperatorPool::GetAllOps(), true);
  HT_LOG_DEBUG << "Topo order: " << _exec_ctx->_global_topo_order;
  bool training = _exec_ctx->_global_topo_order.end() !=
    std::find_if(_exec_ctx->_global_topo_order.begin(),
                 _exec_ctx->_global_topo_order.end(),
                 [](const Operator& op) { return is_optimizer_update_op(op); });
  if (parallel) {
    // 1. 推导每个tensor的states/order
    DeduceDistributedStates(_exec_ctx->_global_topo_order);
    // 2. 获取local topo
    _exec_ctx->_local_topo_order =
      get_local_nodes(_exec_ctx->_global_topo_order, _exec_ctx->_local_device);
    // 3. 将local_topo中的comm_op替换为具体的通信op(集合通信 or 与特定devices特定data slice的通信), 用于后续runtime时进行tensor的实际转换
    SubstituteCommOp(_exec_ctx->_local_topo_order);

    if (_exec_ctx->is_pipeline_parallel()) {
      HT_LOG_DEBUG << "Topo order of pipeline stage: "
                   << _exec_ctx->_local_topo_order;
      HT_ASSERT(!training || !losses.empty())
        << "Must provide loss when training in pipeline parallel";
    }
  } else {
    _exec_ctx->_local_topo_order = _exec_ctx->_global_topo_order;
  }
  _exec_ctx->_losses = losses;
  _exec_ctx->_training = training;
}

NDArrayList DARExecutor::Run(const TensorList& fetches,
                             const FeedDict& feed_dict,
                             const int num_micro_batches) {
  auto sub_exec = GetOrCreateSubExecutor(fetches);
  if (num_micro_batches > 1) {
    HT_ASSERT(_exec_ctx->is_pipeline_parallel())
      << "num_micro_batches > 1 only allowed in the pipeline parallelism!";
  }
  return sub_exec->Run(fetches, feed_dict, num_micro_batches);
}

bool DARExecutor::PlaceDevices(const OpList& topo_order) {
  HT_LOG_TRACE << "Device placement for topo: " << topo_order;
  // If any device_group passed to executor or any operator
  // contains 2 or more devices, then we should run in parallel.
  bool parallel = _exec_ctx->_device_group.num_devices() > 1;
  if (!parallel) {
    for (const auto& op : topo_order) {
      if (op->device_group().num_devices() > 1) {
        parallel = true;
        break;
      }
    }
  }
  // TODO: return the inserted ops during mapping and placement
  // so that we need not to call the TopoSort again
  // 做两轮map: 1. 标注op所在的device placement_group + op->output_tensor的placement_group/device_num(new); 2. 标注local device上的op的device placement + op->output_tensor的placement
  // 注：通过states/partial/duplicate来确定每个placement device上某op计算出的tensor, 与op所在的整个placement device group中的逻辑大tensor之间的关系; tensor的placement group和placement在这里完成, 具体的states/partial/duplicate后面需要再加一个遍历来做
  // 比如纯dp中, 初始状态给4个device都feed了一份{x,y}, 实际上可以看作是PlaceholderOp把原始的大tensor横向切分为4份分配给每个device, 即states[0]=4, 该op在每个device的output都是逻辑上大tensor的横向切分的4份中的1份 
  if (parallel) {
    MapOpsToParallelDevices(topo_order, _exec_ctx->_device_group); // for global info: op->placement_group + tensor->placement_group
    OpList updated_topo_order =
      TopoSort(ExtendSubgraphWithCommunicationNodes(topo_order));
    OpList local_topo_order =
      get_local_nodes(updated_topo_order, _exec_ctx->_local_device);
    PlaceToLocalDevice(local_topo_order, _exec_ctx->_local_device); // for local compute: op->placement + tensor->placement
  } else {
    PlaceToLocalDevice(topo_order, _exec_ctx->_local_device);
  }
  // TODO: return the specific parallel strategy
  return parallel;
}

// tensor1 = op1(xxx)
// tensor2 = comm_op(tensor1, dst_distributed_state) // distributed_state是给定的op2 input所需的切分状态
// tensor3 = op2(tensor2) 
// 推导所有tensor的distributed_states
void DARExecutor::DeduceDistributedStates(const OpList& topo_order) {
  for (auto& op : topo_order) {
    if (is_comm_op(op)) { // 注: 这一部分可以写在CommOp的DoDeduceDistributedStates()里
      CommOp& op = reinterpret_cast<CommOp&>(op);
      auto src_distributed_states = op->input(0)->get_distributed_states();
      auto dst_distributed_states = op->get_dst_distributed_states();
      HT_ASSERT(src_distributed_states.is_valid() && dst_distributed_states.is_valid() && 
                src_distributed_states.get_device_num() == dst_distributed_states.get_device_num())
                << "cannot convert src distributed states to unpaired dst distributed states!";
      op->output(0)->get_distributed_states().set_distributed_states(dst_distributed_states);
    } else if (is_data_loader_op(op)) {
      ; // 待定
    } else if (is_placeholder_op(op) || is_variable_op(op)) { // placeholder_op和variable_op的output tensor的distributed attributes必须要在创建时手动指定
      HT_ASSERT(op->output(0)->get_distributed_states().is_valid())
                << "tensor from placeholder_op/variable_op must initialize distributed attributes first!";
    } else { // computing ops, 需要根据input tensor的distributed attributes来推导output tensor的distributed attributes
      op->DoDeduceDistributedStates(); // 每个op需要重载实现
    }
  }
}

// tensor1 = op1(xxx)
// tensor2 = comm_op(tensor1, distributed_state) // distributed_state是给定的op2 input所需的切分状态
// tensor3 = op2(tensor2) 
void DARExecutor::SubstituteCommOp(const OpList& local_topo_order) {
  // comm_op的placement和placement_group等价于tensor1的, 所以comm_op在local_topo上意味着tensor1的placement==local_device
  for (auto& op : local_topo_order) {
    if (is_comm_op(op)) {
      CommOp& op = reinterpret_cast<CommOp&>(op);
      uint64_t comm_type = op->get_comm_type();
      Tensor result;
      if (comm_type == ALL_REDUCE_OP) {
        AllReduceOp all_reduce_op(
          op->input(0),
          OpMeta().set_device_group(op->device_group()).set_name(op->input(0)->name() + "_AllReduce"));
        all_reduce_op->MapToParallelDevices(op->device_group());
        result = all_reduce_op->output(0); // result就是tensor2
      } else if (comm_type == ALL_GATHER_OP) {
        ;
      } else if (comm_type == REDUCE_SCATTER_OP) {
        ;
      } else if (comm_type == BROADCAST_OP) {
        ;
      } else if (comm_type == REDUCE_OP) {
        ;
      } else if (comm_type == P2P_OP) {
        ; // 分为两部分: 1. 从local_device发送数据给需要的device 2. 从目标devices中接收需要的数据到local_device
        int32_t device_index = 0;
        CrossSend({}, {}, 0, false, device_index, op); // 注: partial的情况还要再想一下
        HT_ASSERT(device_index == _exec_ctx->device_group().num_devices()) << "cross send error!";
        device_index = 0;
        result = CrossReceive(0, device_index, op);
        HT_ASSERT(device_index == _exec_ctx->device_group().num_devices()) << "cross receive error!";        
      }

      // 找到所有消费tensor2的op, 替换对应的input: 从占位符comm_op的output(0)替换为真正通信算子得到的result
      for (size_t i = 0; i < op->output(0)->num_consumers(); i++) {
        for (size_t j = 0; j < op->output(0)->consumer(i)->num_inputs(); j++) {
          if (op->output(0)->consumer(i)->input(j)->id() == op->output(0)->id()) {
            op->output(0)->consumer(i)->ReplaceInput(j, result); // op->ReplaceInput()原先是protected, 这里暂时挪到public来              
          }
        }
      }      
    }
  }
}

// 两个问题: 1. cross send的cur_state_index记录的是local device在prev states/order下分到了哪些数据, 通过device_index来记录local device需要发送的目标devices,
// 并将split或原数据发送给它们; cross receive的cur_state_index记录的是local device在target states/order下需要的是哪些数据, 通过device_index记录这些所需数据来源
// 的device, 将它们传来的数据进行sum/concatenate作为distributed tensor在local device转换后的part result tensor
// 2. device_index记录的是device_group中按照order排序后的device的index? 比如states={-1:2, 0:2}, order=[-1, 0], device_group=[0,1,2,3], 
// 而order=[0, -1], device_group=[0, 2, 1, 3], 如果device_index=2的话, 在这两种情况下的目标device应该分别是device 2和device 1?
Tensor DARExecutor::CrossReceive(int32_t depth, int32_t& device_index, CommOp& op) {
  auto prev_distributed_states = op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  auto prev_order = prev_distributed_states.get_order();
  auto loop_sizes = prev_distributed_states.get_loop_sizes();

  auto target_distributed_states = op->get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  // 这里暂时默认prev和target的distributed states的placement group是同一个
  // auto cur_state_index = target_distributed_states.map_device_to_state_index(DeviceToWorldRank(_exec_ctx->local_device())); // local device需要的是哪些数据
  // auto cur_state_index = target_distributed_states.map_device_to_state_index(_exec_ctx->local_device().index()); // local device需要的是哪些数据
  auto cur_state_index = target_distributed_states.map_device_to_state_index(prev_distributed_states.get_placement_group().get_index(_exec_ctx->local_device())); // local device需要的是哪些数据

  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };

  Tensor result;
  // cur_state_index存的是local device需要的是哪些数据, 最终的result是从device_index对应的device中concatenate/sum获取而来的
  if (depth == prev_order.size()) {
    // result = general_receiving(prev_distributed_states.get_placement_group(), 
    //                            target_distributed_states.get_placement_group(), device_index);
    device_index += 1;            
  } else {
    auto cur_dim = prev_order[depth];
    if (cur_dim == -2) { // partial
      TensorList part_result_list;
      for (size_t i = 0; i < prev_partial; i++) {
        auto part_result = CrossReceive(depth+1, device_index, op);
        part_result_list.push_back(part_result);
      }
      // result = sum_op(part_result_list); // sum_op的placement/placement_group还需要设置
    } else if (cur_dim == -1) {
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, op);
        device_index += ((target_duplicate - cur_st) * multiple - 1) * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, op);
        device_index += (target_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_states()[cur_dim];
      auto tar_st = target_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        if (multiple == 1) {
          result = CrossReceive(depth+1, device_index, op);
        } else {
          TensorList part_result_list;
          for (size_t i = 0; i < multiple; i++) {
            auto part_result = CrossReceive(depth+1, device_index, op);
            part_result_list.push_back(part_result);
          }
          // result = concatenate_op(part_result_list, axis=cur_dim); // 在cur_dim这一维做合并, concatenate_op的placement/placement_group还需要设置 
        }
        device_index += (tar_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, op);
        device_index += (tar_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    }
  }
  
  return result;
}

void DARExecutor::CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                std::unordered_map<int32_t, int32_t> split_target_state,
                int32_t depth, bool need_split, int32_t& device_index, CommOp& op) {
  // basic info
  auto prev_distributed_states = op->input(0)->get_distributed_states();
  HT_ASSERT(_exec_ctx->local_device() == prev_distributed_states.get_placement()) 
            << "local device must be the comm_op input tensor's placement!";
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  auto cur_state_index = prev_distributed_states.map_device_to_state_index(prev_distributed_states.get_placement_index()); // 根据local_device index和order确定local_device拥有的是tensor1的哪部分数据

  auto target_distributed_states = op->get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  auto target_order = target_distributed_states.get_order();
  auto loop_sizes = target_distributed_states.get_loop_sizes();                  
  
  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };

  auto get_keys = [](std::unordered_map<int32_t, int32_t> map) -> std::vector<int32_t> {
    std::vector<int32_t> keys; 
    keys.reserve(map.size());
    for (auto kv : map) {
      keys.push_back(kv.first);
    }
    return keys;
  };

  // cross send part
  if (depth == target_order.size()) {
    Tensor send_part;
    if (need_split) {
      auto keys = get_keys(split_target_state);
      std::vector<int32_t> indices, splits;
      indices.reserve(keys.size()); splits.reserve(keys.size());
      for (auto key : keys) {
        indices.push_back(split_cur_state[key]);
        splits.push_back(split_target_state[key]);
      }
      // split_op: 把tensor1在keys这些dimension上按照splits[key]份数切分, 并取出第indices[key]份, 作为要send的数据切片 
      // send_part = split_op(op->input(0), keys, indices, splits)->output(0); // split_op的placement/placement_group还需要设置
    } else {
      // 如果不需要split, 则发送整个tensor1
      send_part = op->input(0);
    }
    // 把send_part发送给target device group里的device_index这个device
    // 这里到时候还需要明确下: device_index应该是根据order排序后的device group里的index ?
    // general_sending(send_part, prev_distributed_states.get_placement_group(), 
    //                 target_distributed_states.get_placement_group(), device_index);
    device_index += 1;
  } else {
    auto cur_dim = target_order[depth];
    if (cur_dim < 0) {
      HT_ASSERT(cur_dim == -1) << "Target distributed states must not enable partial!";
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        if (cur_st % multiple != 0) {
          HT_LOG_INFO << "prev_duplicate % target_duplicate == 0 but cur_st % multiple != 0, "
          << "cur_st: "<< cur_st << ", multiple: " << multiple; 
          return;
        }
        device_index += cur_st / multiple * loop_sizes[depth];
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, op);
        device_index += (prev_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, op);
        }
        device_index += (prev_duplicate - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_LOG_INFO << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      auto tar_st = target_distributed_states.get_states()[cur_dim];
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        split_cur_state[cur_dim] = 0;
        split_target_state[cur_dim] = 1;
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, op);
        device_index += (pre_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          split_cur_state[cur_dim] = i;
          split_target_state[cur_dim] = multiple; 
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, op);
        }
        device_index += (pre_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_LOG_INFO << "cannot support!";
      }
    }
  }
}

std::shared_ptr<DARSubExecutor>
DARExecutor::GetOrCreateSubExecutor(const TensorList& fetches) {
  TIK(get_or_create);
  std::shared_ptr<DARSubExecutor> sub_exec = nullptr;
  // Lookup sub_executors by fetch_ids.
  TensorIdList fetch_ids(fetches.size());
  std::transform(fetches.begin(), fetches.end(), fetch_ids.begin(),
                 [](const Tensor& tensor) { return tensor->id(); });
  auto fetches_it = _fetches_to_sub_executors.find(fetch_ids);
  if (fetches_it == _fetches_to_sub_executors.end()) {
    // Lookup sub_executors by topo_ids. Here we sort fetches to
    // ensure that the topo order is deterministic.
    TensorList sorted_fetches(fetches.size());
    std::partial_sort_copy(
      fetches.begin(), fetches.end(), sorted_fetches.begin(),
      sorted_fetches.end(),
      [](const Tensor& a, const Tensor& b) { return a->id() < b->id(); });
    auto topo_order = TopoSort(sorted_fetches, true);
    OpList local_fw_topo, local_bw_topo;
    if (_exec_ctx->is_pipeline_parallel()) {
      auto parts = disentangle_forward_and_backward_nodes(
        topo_order, _exec_ctx->_losses, true);
      local_fw_topo =
        get_local_nodes(std::get<0>(parts), _exec_ctx->_local_device);
      local_bw_topo =
        get_local_nodes(std::get<1>(parts), _exec_ctx->_local_device);
      topo_order.clear();
      topo_order.reserve(local_fw_topo.size() + local_bw_topo.size());
      topo_order.insert(topo_order.end(), local_fw_topo.begin(),
                        local_fw_topo.end());
      topo_order.insert(topo_order.end(), local_bw_topo.begin(),
                        local_bw_topo.end());
    }
    // get or create a sub_executor according to the topo order
    OpIdList topo_order_ids(topo_order.size());
    std::transform(topo_order.begin(), topo_order.end(), topo_order_ids.begin(),
                   [](const Operator& op) { return op->id(); });
    auto topo_it = _topo_to_sub_executors.find(topo_order_ids);
    if (topo_it == _topo_to_sub_executors.end()) {
      // create a sub_executor
      if (_exec_ctx->is_pipeline_parallel()) {
        sub_exec = std::make_shared<PipelineDARSubExecutor>(
          _exec_ctx, topo_order, local_fw_topo, local_bw_topo);
      } else {
        sub_exec =
          std::make_shared<DefaultDARSubExecutor>(_exec_ctx, topo_order);
      }
      _topo_to_sub_executors.insert({topo_order_ids, sub_exec});
      TOK(get_or_create);
      HT_LOG_DEBUG << "Create SubExecutor for fetches " << fetches << " cost "
                   << COST_MICROSEC(get_or_create) / 1000.0 << " ms";
    } else {
      // reuse the sub_executor
      sub_exec = topo_it->second;
    }
    _fetches_to_sub_executors.insert({fetch_ids, sub_exec});
  } else {
    sub_exec = fetches_it->second;
  }

  return sub_exec;
}

DARSubExecutor::DARSubExecutor(std::shared_ptr<DARExecutionContext> exec_ctx,
                               const OpList& topo_order)
: _exec_ctx(exec_ctx), _topo_order(topo_order) {
  std::unordered_set<OpId> ops;
  ops.reserve(_topo_order.size());
  size_t num_out_edges = 0;
  for (auto& op : _topo_order) {
    ops.insert(op->id());
    num_out_edges += op->num_outputs();
    if (is_variable_op(op)) {
      _variable_ops.push_back(op);
    } else if (is_placeholder_op(op)) {
      _placeholder_ops.push_back(op);
    } else if (is_data_loader_op(op)) {
      _data_loader_ops.push_back(op);
    } else {
      _computing_ops.push_back(op);
    }
  }

  _edge_out_degrees.reserve(num_out_edges);
  for (auto& op : _topo_order) {
    for (auto& output : op->outputs())
      _edge_out_degrees[output->id()] = 0;
    for (auto& input : op->inputs())
      _edge_out_degrees[input->id()]++;
  }
}

NDArrayList DefaultDARSubExecutor::Run(const TensorList& fetches,
                                       const FeedDict& feed_dict,
                                       const int num_micro_batches) {
  Tensor2NDArrayMap edge2arr;
  std::unordered_map<TensorId, int> edge2degrees = _edge_out_degrees;
  std::unordered_set<OpId> fetch_ids;
  fetch_ids.reserve(fetches.size());
  for (const auto& fetch : fetches)
    fetch_ids.insert(fetch->id());

  // get feed in values
  for (const auto& kv : feed_dict) {
    // TODO: transfer H2D if needed
    // TODO: check shape & dtype
    edge2arr[kv.first] = kv.second;
  }
  for (auto& op : _placeholder_ops) {
    HT_ASSERT(edge2arr.find(op->output(0)->id()) != edge2arr.end())
      << "Cannot find values for op \"" << op->name() << "\" in feed_dict";
  }

  // get variable values
  for (auto& op : _variable_ops) {
    VariableOp& var = reinterpret_cast<VariableOp&>(op);
    edge2arr[op->output(0)->id()] = var->data();
  }

  // get dataloader values
  for (auto& op : _data_loader_ops) {
    DataloaderOp& var = reinterpret_cast<DataloaderOp&>(op);
    int idx = 0;
    for (auto it = var->dataloaders().begin(); it != var->dataloaders().end();
         ++it, ++idx) {
      edge2arr[op->output(idx)->id()] = it->second->get_arr();
    }
  }

  RuntimeContext runtime_ctx;

  // compute
  for (auto& op : _computing_ops) {
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT(it != edge2arr.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      input_vals.push_back(it->second);
      if ((--edge2degrees[in_edge->id()]) == 0 &&
          fetch_ids.find(in_edge->id()) == fetch_ids.end()) {
        edge2arr.erase(in_edge->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      if (edge2degrees[out_edge->id()] > 0 ||
          fetch_ids.find(out_edge->id()) != fetch_ids.end()) {
        edge2arr[out_edge->id()] = output_vals[i];
      }
    }
  }

  // get results
  NDArrayList results;
  for (const auto& fetch : fetches) {
    auto& op = fetch->producer();
    if (_exec_ctx->is_pipeline_parallel() &&
        op->placement() != _exec_ctx->local_device()) {
      results.push_back(NDArray());
      continue;
    }
    if (is_variable_op(op) || is_placeholder_op(op) || is_data_loader_op(op)) {
      results.push_back(edge2arr[fetch->id()]);
    } else {
      op->Sync();
      auto it = edge2arr.find(fetch->id());
      if (it != edge2arr.end())
        results.push_back(it->second);
      else
        results.push_back(NDArray());
    }
  }
  return results;
}

PipelineDARSubExecutor::PipelineDARSubExecutor(
  std::shared_ptr<DARExecutionContext> exec_ctx, const OpList& topo_order,
  const OpList& fw_topo_order, const OpList& bw_topo_order)
: DARSubExecutor(exec_ctx, topo_order),
  _fw_topo_order(fw_topo_order),
  _bw_topo_order(bw_topo_order) {
  // fw/bw compute ops
  for (auto& op : _fw_topo_order) {
    if (!is_variable_op(op) && !is_placeholder_op(op) &&
        !is_data_loader_op(op)) {
      _fw_computing_ops.push_back(op);
    }
  }
  for (auto& op : _bw_topo_order) {
    if (!is_variable_op(op) && !is_placeholder_op(op) &&
        !is_data_loader_op(op)) {
      _bw_computing_ops.push_back(op);
    }
  }
  // gradient ops
  OpList allreduce_ops;
  OpList update_ops;
  for (auto& op : _bw_computing_ops) {
    if (is_all_reduce_op(op)) {
      allreduce_ops.push_back(op);
    }
    if (is_optimizer_update_op(op)) {
      update_ops.push_back(op);
    }
  }
  // gradient consumer ops
  if (allreduce_ops.size() > 0) {
    _gradient_ops = allreduce_ops;
  } else {
    _gradient_ops = update_ops;
  }
  for (auto& gradient_op : _gradient_ops) {
    std::queue<Tensor> q;
    if (is_optimizer_update_op(gradient_op)) {
      q.push(gradient_op->out_dep_linker());
    } else {
      q.push(gradient_op->output(0));
    }
    while (!q.empty()) {
      auto& output = q.front();
      q.pop();
      for (size_t i = 0; i < output->num_consumers(); i++) {
        auto& consumer_op = output->consumer(i);
        _gradient_consumer_ops.push_back(consumer_op);
        for (size_t j = 0; j < consumer_op->num_outputs(); j++) {
          q.push(consumer_op->output(j));
        }
        if (is_optimizer_update_op(consumer_op)) {
          q.push(consumer_op->out_dep_linker());
        }
      }
    }
  }
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<int, std::vector<std::pair<bool, int>>>
PipelineDARSubExecutor::generate_gpipe_schedule(int num_stages,
                                                int num_micro_batches) {
  std::unordered_map<int, std::vector<std::pair<bool, int>>> schedule;
  // inference time: for only forward
  if (_bw_computing_ops.size() == 0) {
    for (int stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, int>> tasks;
      tasks.reserve(num_micro_batches);
      for (int step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (int stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, int>> tasks;
    tasks.reserve(2 * num_micro_batches);
    for (int step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({true, step_id});
    }
    for (int step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({false, step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<int, std::vector<std::pair<bool, int>>>
PipelineDARSubExecutor::generate_pipedream_flush_schedule(
  int num_stages, int num_micro_batches) {
  HT_ASSERT(num_micro_batches >= num_stages)
    << "num_micro_batches must bigger than num_stages in pipedream-flush!";
  std::unordered_map<int, std::vector<std::pair<bool, int>>> schedule;
  // inference time: for only forward
  if (_bw_computing_ops.size() == 0) {
    for (int stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, int>> tasks;
      tasks.reserve(num_micro_batches);
      for (int step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (int stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, int>> tasks;
    tasks.reserve(2 * num_micro_batches);
    int num_warmup_microbatches = num_stages - stage_id - 1;
    int num_microbatches_remaining =
      num_micro_batches - num_warmup_microbatches;
    // 1. warmup
    for (int step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({true, step_id});
    }
    // 2. 1F1B
    for (int step_id = 0; step_id < num_microbatches_remaining; step_id++) {
      tasks.push_back({true, num_warmup_microbatches + step_id});
      tasks.push_back({false, step_id});
    }
    // 3. cooldown
    for (int step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({false, num_microbatches_remaining + step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

void PipelineDARSubExecutor::compute_fn(
  OpList& compute_ops, Tensor2NDArrayMap& edge2arr,
  std::unordered_map<TensorId, int>& edge2degrees,
  std::unordered_map<TensorId, NDArray>& grad_accumulation,
  bool grad_accumulation_finished, std::unordered_set<OpId>& fetch_ids,
  RuntimeContext& runtime_ctx) {
  for (auto& op : compute_ops) {
    // grad accmulation for udpate_op and all_reduce_op just before update_op
    TensorId grad_id = -1;
    if (is_all_reduce_op(op) &&
        is_optimizer_update_op(op->output(0)->consumer(0))) {
      grad_id = op->input(0)->id();
    }
    if (is_optimizer_update_op(op) &&
        !is_all_reduce_op(op->input(1)->producer())) {
      grad_id = op->input(1)->id();
    }
    if (grad_id != -1) {
      if (grad_accumulation.find(grad_id) == grad_accumulation.end()) {
        grad_accumulation[grad_id] = edge2arr[grad_id];
      } else {
        grad_accumulation[grad_id] =
          NDArray::add(grad_accumulation[grad_id], edge2arr[grad_id]);
      }
      if (grad_accumulation_finished) {
        edge2arr[grad_id] = grad_accumulation[grad_id];
      } else {
        continue;
      }
    } else if (!grad_accumulation_finished) {
      bool is_consumer_op = _gradient_consumer_ops.end() !=
        std::find_if(_gradient_consumer_ops.begin(),
                     _gradient_consumer_ops.end(), [&](Operator& consumer_op) {
                       return consumer_op->id() == op->id();
                     });
      if (is_consumer_op) {
        continue;
      }
    }

    // compute
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT(it != edge2arr.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      input_vals.push_back(it->second);
      if ((--edge2degrees[in_edge->id()]) == 0 &&
          fetch_ids.find(in_edge->id()) == fetch_ids.end()) {
        edge2arr.erase(in_edge->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      if (edge2degrees[out_edge->id()] > 0 ||
          fetch_ids.find(out_edge->id()) != fetch_ids.end()) {
        edge2arr[out_edge->id()] = output_vals[i];
      }
    }
  }
}

NDArrayList PipelineDARSubExecutor::Run(const TensorList& fetches,
                                        const FeedDict& feed_dict,
                                        const int num_micro_batches) {
  std::vector<Tensor2NDArrayMap> edge2arr_list(
    num_micro_batches); // edge2arr for m micro batches
  std::vector<std::unordered_map<TensorId, int>> edge2degrees_list(
    num_micro_batches,
    _edge_out_degrees); // // edge2degrees for m micro batches
  std::vector<RuntimeContext> runtime_ctx_list;
  runtime_ctx_list.reserve(
    num_micro_batches); // // runtimectx for m micro batches
  std::unordered_map<TensorId, NDArray> grad_accumulation;
  grad_accumulation.reserve(
    _variable_ops.size()); // for weights grad accumulation
  std::unordered_set<OpId> fetch_ids;
  fetch_ids.reserve(fetches.size());

  for (int i = 0; i < num_micro_batches; i++)
    runtime_ctx_list.push_back(RuntimeContext());

  for (const auto& fetch : fetches)
    fetch_ids.insert(fetch->id());

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 1. init[start]";
  // 1. init
  // get feed in values
  for (const auto& kv : feed_dict) {
    // TODO: transfer H2D if needed
    // TODO: check shape & dtype
    auto micro_batches = NDArray::split(kv.second, num_micro_batches);
    for (int i = 0; i < num_micro_batches; i++) {
      edge2arr_list[i][kv.first] = micro_batches[i];
    }
  }
  for (auto& op : _placeholder_ops) {
    for (auto& edge2arr : edge2arr_list) {
      HT_ASSERT(edge2arr.find(op->output(0)->id()) != edge2arr.end())
        << "Cannot find values for op \"" << op->name() << "\" in feed_dict";
    }
  }

  // get variable values
  for (auto& op : _variable_ops) {
    VariableOp& var = reinterpret_cast<VariableOp&>(op);
    for (auto& edge2arr : edge2arr_list) {
      edge2arr[op->output(0)->id()] = var->data();
    }
  }

  // get dataloader values
  for (auto& op : _data_loader_ops) {
    DataloaderOp& var = reinterpret_cast<DataloaderOp&>(op);
    int idx = 0;
    for (auto it = var->dataloaders().begin(); it != var->dataloaders().end();
         ++it, ++idx) {
      for (auto& edge2arr : edge2arr_list) {
        edge2arr[op->output(idx)->id()] = it->second->get_arr();
      }
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 1. init[end]";

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 2. compute[start]";
  // 2. compute
  // TODO: pp stage should support different dp degrees
  int dp = _exec_ctx->global_topo_order()[0]
             ->placement_group()
             .num_devices(); // data parallel degree
  int num_stages =
    _exec_ctx->device_group().num_devices() / dp; // get stage num
  auto schedule = generate_pipedream_flush_schedule(
    num_stages,
    num_micro_batches); // get task schedule table for pipedream-flush
  // auto schedule = generate_gpipe_schedule(num_stages, num_micro_batches); //
  // // get task schedule table for gpipe
  auto& tasks = schedule[_exec_ctx->local_device().index() /
                         dp]; // get tasks for current device
  for (std::size_t i = 0; i < tasks.size(); i++) {
    auto& task = tasks[i];
    bool is_forward = task.first;
    int micro_batch_id = task.second;
    auto& edge2arr = edge2arr_list[micro_batch_id];
    auto& edge2degrees = edge2degrees_list[micro_batch_id];
    auto& runtime_ctx = runtime_ctx_list[micro_batch_id];
    if (is_forward) {
      compute_fn(_fw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 false, fetch_ids, runtime_ctx);
    } else if (i < tasks.size() - 1) {
      compute_fn(_bw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 false, fetch_ids, runtime_ctx);
    } else {
      compute_fn(_bw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 true, fetch_ids, runtime_ctx);
    }
    if (is_forward) {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": [micro batch "
                   << micro_batch_id << ": forward]";
    } else {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": [micro batch "
                   << micro_batch_id << ": backward]";
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 2. compute[end]";

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 3. get results[start]";
  // 3. get results
  NDArrayList results;
  for (const auto& fetch : fetches) {
    auto& op = fetch->producer();
    if (_exec_ctx->is_pipeline_parallel() &&
        op->placement() != _exec_ctx->local_device()) {
      results.push_back(NDArray());
      continue;
    }
    if (is_variable_op(op)) {
      results.push_back(edge2arr_list[num_micro_batches - 1][fetch->id()]);
      HT_LOG_DEBUG
        << _exec_ctx->local_device() << ": fetch " << fetch
        << " is varibale op's output, get result in last micro batch updated, shape: "
        << results.back()->shape();
    } else if (is_placeholder_op(op) || is_data_loader_op(op)) {
      NDArrayList result;
      result.reserve(num_micro_batches);
      for (auto& edge2arr : edge2arr_list) {
        result.push_back(edge2arr[fetch->id()]);
      }
      results.push_back(NDArray::cat(result));
      HT_LOG_DEBUG
        << _exec_ctx->local_device() << ": fetch " << fetch
        << " is placeholder/data_loader op's output, cat result in all micro batches, shape: "
        << results.back()->shape();
    } else {
      op->Sync();
      // case 1
      auto it_grad = grad_accumulation.find(fetch->id());
      if (it_grad != grad_accumulation.end()) {
        results.push_back(it_grad->second);
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is allreduce/update op's input, get result directly in grad_accumulation, shape: "
          << results.back()->shape();
        continue;
      }
      // case 2
      if ((is_all_reduce_op(op) &&
           is_optimizer_update_op(op->output(0)->consumer(0))) ||
          is_optimizer_update_op(op)) {
        results.push_back(edge2arr_list[num_micro_batches - 1][fetch->id()]);
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is allreduce/update op's output, get result in last micro batch updated, shape: "
          << results.back()->shape();
        continue;
      }
      // case 3
      auto it = edge2arr_list[num_micro_batches - 1].find(fetch->id());
      if (it != edge2arr_list[num_micro_batches - 1].end()) {
        NDArrayList result;
        result.reserve(num_micro_batches);
        for (auto& edge2arr : edge2arr_list) {
          result.push_back(edge2arr[fetch->id()]);
        }
        results.push_back(NDArray::cat(result));
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is common compute op's output, cat result in all micro batches, shape: "
          << results.back()->shape();
      } else {
        results.push_back(NDArray());
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": fetch " << fetch
                     << " is common compute op's output, but return NDArray()";
      }
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 3. get results[end]";
  return results;
}

} // namespace execution
} // namespace hetu
