#include "hetu/graph/headers.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/core/device.h"
#include "hetu/core/dtype.h"
#include "hetu/core/ndarray_meta.h"
#include "hetu/core/stream.h"
#include "hetu/common/timing.h"
#include <nvml.h>
#include <nccl.h>

namespace hetu {
namespace graph {

std::ostream& operator<<(std::ostream& os, const SwitchExecGraph& switcher) {
  os << "switch_exec_graph(" << switcher.SwitchGraphPair().first->name() << ", " 
    << switcher.SwitchGraphPair().second->name() << ")";
  return os;
}

template<typename Key, typename Value>
static std::unordered_set<Key> KeysUnion(const std::unordered_map<Key, Value>& map1, const std::unordered_map<Key, Value>& map2)
{
  std::unordered_set<Key> result;
  for (const auto& pair : map1) {
    result.insert(pair.first);
  }
  for (const auto& pair : map2) {
    result.insert(pair.first);
  }
  return result;
}

template<typename Key>
static std::unordered_set<Key> KeysUnion(const std::vector<Key>& vec1, const std::vector<Key>& vec2)
{
  std::unordered_set<Key> result;
  for (const auto& key : vec1) {
    result.insert(key);
  }
  for (const auto& key : vec2) {
    result.insert(key);
  }
  return result;
}

template<typename Key>
static std::unordered_set<Key> KeysUnion(const std::unordered_set<Key>& set1, const std::unordered_set<Key>& set2)
{
  std::unordered_set<Key> result;
  std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
  return result;
}

void ParamSlice::AddOwnedSliceInst(const Device& device, const Tensor& tensor) {
  if (!_owned_slice_instances.empty()) {
    const HTShape& shape = _owned_slice_instances[0]->shape();
    const auto shape_size = shape.size();
    HT_ASSERT(shape_size == tensor->shape().size())
      << "the new slice instance shape should be equal to the old slice instance shape, " 
      << "but the new is " << tensor->shape() << " and the old is " << shape;
    for(size_t i = 0; i < shape_size; ++i) {
      HT_ASSERT(shape[i] == tensor->shape(i))
        << "the new slice instance shape should be equal to the old slice instance shape, "  
        << "but the new is " << tensor->shape() << " and the old is " << shape;
    }
  }
  _owned_devices.push_back(device);
  _owned_slice_instances.push_back(tensor);
  _switcher->RecordTensorInfo(tensor, name());
}

void ParamSlice::AddNeededSliceInst(const Device& device, const Tensor& tensor) {
  HT_ASSERT(!_owned_slice_instances.empty())
    << "the slice isn't owned by any devices, "
    << "please ensure you've added a slice instance before";
  const HTShape& shape = _owned_slice_instances[0]->shape();
  const auto shape_size = shape.size();
  HT_ASSERT(shape_size == tensor->shape().size())
    << "the needed slice shape should be equal to the owned slice shape, " 
    << "but the needed is " << tensor->shape() << " and the owned is " << shape;
  for(size_t i = 0; i < shape_size; ++i) {
    HT_ASSERT(shape[i] == tensor->shape(i))
      << "the needed slice shape should be equal to the owned slice shape, " 
      << "but the needed is " << tensor->shape() << " and the owned is " << shape;
  }
  _needed_devices.push_back(device);
  _needed_slice_instances.push_back(tensor);
}

// TODO: 更好的算法
void ParamSlice::ParamSliceComm(Device2DTListPairMap& send_mapping,
                                Device2DTListPairMap& recv_mapping) {
  auto needed_len = _needed_slice_instances.size();
  auto owned_len = _owned_slice_instances.size();
  HT_ASSERT(needed_len == _needed_devices.size() && owned_len == _owned_devices.size())
    << "something wrong with the size";
  for (size_t i = 0; i < needed_len; ++i) {
    auto& needed_device = _needed_devices[i];
    bool already_owned = false;
    size_t already_owned_slice_instance_num = 0;
    // 先扫一遍，如果自己已经有了，那么就不需要通信了
    for (size_t j = 0; j < owned_len; ++j) {
      if (needed_device == _owned_devices[j]) {
        already_owned = true;
        already_owned_slice_instance_num= j;
        break;
      }
    }
    if (already_owned) {
      // 只需要替换即可
      // 不需要记录在send/recv的mapping中
      auto& old_tensor = _needed_slice_instances[i];
      auto& new_tensor = _owned_slice_instances[already_owned_slice_instance_num];
      HT_ASSERT(old_tensor->num_consumers() == 1)
        << "the slice instance should only used once (by a single concatenate op)";
      auto& consumer = old_tensor->consumer(0);
      for (size_t j = 0; j < consumer->num_inputs(); ++j) {
        if (consumer->input(j)->id() == old_tensor->id()) {
          Graph::ReplaceInput(consumer, j, new_tensor);
        }
      }
      HT_LOG_DEBUG_IF(needed_device == hetu::impl::comm::GetLocalDevice())
        << needed_device << ": can reuse the " << name()
        << " param slice instance owned by itself";
    } else {
      // 需要通信
      // 通信关系会记录在send/recv的mapping中
      Tensor send_tensor; // TBD
      Device send_device; // TBD
      auto& recv_tensor = _needed_slice_instances[i];
      auto& recv_device = _needed_devices[i];
      // 不同的算法
      // FCFS/ round-robin
      if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::FCFS
          || _switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN) {
        send_tensor = _owned_slice_instances[_round_robin];
        send_device = _owned_devices[_round_robin];
        // 更新轮询次数
        if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN) {
          _round_robin++;
        }
        if (_round_robin == _owned_slice_instances.size()) {
          _round_robin = 0;
        }
      } 
      // 按照已经通信的次数进行greedy（即选取已通信中p2p次数最小的）
      if (_switcher->_algorithm_level == SWITCH_ALGORITHM_LEVEL::GREEDY) {
        std::pair<Device, Device> best_p2p;
        size_t best_send_num;
        auto& p2p_val_mapping = _switcher->_p2p_val_mapping;
        size_t min_val = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < owned_len; ++j) {
          // p2p是双向通路
          // 规定device编号小的在前
          std::pair<Device, Device> p2p;
          if (_owned_devices[j] < recv_device) {
            p2p = std::make_pair(_owned_devices[j], recv_device);
          } else {
            p2p = std::make_pair(recv_device, _owned_devices[j]);
          }
          auto it = p2p_val_mapping.find(p2p);
          // 相当于通信了0次
          if (it == p2p_val_mapping.end()) {
            best_p2p = p2p;
            best_send_num = j;
            break;
          }
          // 选择通信次数最小的
          if (it->second < min_val) {
            min_val = it->second;
            best_p2p = p2p;
            best_send_num = j;
          }
        }
        // 更新p2p_val_mapping
        p2p_val_mapping[best_p2p] += 1;
        send_tensor = _owned_slice_instances[best_send_num];
        send_device = _owned_devices[best_send_num];
      }
      // 建立通信关系
      auto recv_it = recv_mapping.find(recv_device);
      auto send_it = send_mapping.find(send_device);
      HT_ASSERT(send_it != send_mapping.end() && recv_it != recv_mapping.end())
        << "device is not recorded in the send/recv mapping";
      recv_it->second.first.push_back(send_device);
      recv_it->second.second.push_back(recv_tensor);
      send_it->second.first.push_back(recv_device);
      send_it->second.second.push_back(send_tensor);
      HT_LOG_DEBUG_IF(send_device == hetu::impl::comm::GetLocalDevice())
        << send_device << ": will send the " << name()
        << " param slice instance to " << recv_device;
    }
  }
}

// 遍历ParamBlock中的每个ParamSlice
// 找到最优的ParamSliceInst的通信策略
// TODO: 更好的算法
void ParamBlock::ParamBlockComm(Device2DTListPairMap& send_mapping,
                                Device2DTListPairMap& recv_mapping) {
  // auto param_slices_size = _param_slices.size();
  for (auto& param_slice_ptr : _param_slices) {
    param_slice_ptr->ParamSliceComm(send_mapping, recv_mapping);
  }
}

// 递归地为ParamBlock创建所有的ParamSlices
void SwitchExecGraph::CreateParamBlock(ParamBlock& block,
                                      std::vector<int32_t>& slice_num, 
                                      const TensorName& block_name,
                                      int32_t dim) {
  const auto& block_shape = block.BlockShape();
  if (dim == block_shape.size()) {
    block.GetParamSlices().emplace_back(std::make_shared<ParamSlice>(block_name, slice_num, this));
    return;
  }
  for (int32_t i = 0; i < block_shape[dim]; ++i) {
    slice_num[dim] = i;
    CreateParamBlock(block, slice_num, block_name, dim + 1);
  }
}

// 作为发送端
// 将ParamBlock划分成OwnedParamSlice（抽象）
// 切分param成ParamSliceInstance（实际的tensor）
void SwitchExecGraph::MakeAllParamSlices(const Tensor& param, ParamBlock& block, 
                                    const Device& device, const DeviceGroup& group,
                                    std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                                    const std::unordered_map<int32_t, int32_t>& state,
                                    const std::vector<int32_t>& multiple, int32_t dim) {
  if (dim == multiple.size()) {
    auto& param_slice = block.GetParamSlice(slice_num);
    HTShape indices(slice_relative_num.begin(), slice_relative_num.end()); // int32_t -> int64_t
    HTShape splits(multiple.begin(), multiple.end()); // int32_t -> int64_t
    // 都先进行split
    auto split_output = MakeSplitOp(param, indices, splits, OpMeta().set_is_deduce_states(false));
    auto& split_op = split_output->producer();
    // 其他device上生成的不需要map placement_group和placement
    if (hetu::impl::comm::GetLocalDevice() == device) { 
      split_op->MapToParallelDevices(group);
      split_op->Instantiate(device, kComputingStream);
    }
    // dup会导致一个param_slice对应多个slice_instance
    // 这也是这个优化问题之所以这么复杂的原因
    param_slice->AddOwnedSliceInst(device, std::move(split_output));
    return;
  } 
  int32_t basic_slice_num = 0;
  auto it = state.find(dim);
  if (it != state.end()) {
    basic_slice_num = it->second * multiple[dim];
  }
  for (int32_t i = 0; i < multiple[dim]; ++i) {
    slice_num[dim] = basic_slice_num + i;
    slice_relative_num[dim] = i;
    MakeAllParamSlices(param, block, device, group, slice_num, slice_relative_num, state, multiple, dim + 1);
  }                            
}

// 作为接收端
// 将ParamBlock划分成NeededParamSlice（抽象）
// 合并ParamSliceInstance成param（实际的tensor）
Tensor SwitchExecGraph::MergeAllParamSlices(const Tensor& param, ParamBlock& block, 
                                    const Device& device, const DeviceGroup& group,
                                    std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                                    const std::unordered_map<int32_t, int32_t>& state,
                                    const std::vector<int32_t>& multiple, int32_t dim) {
  if (dim == multiple.size()) {
    auto& param_slice = block.GetParamSlice(slice_num);
    const auto& owned_slice_instance = param_slice->OwnedSliceInst(0);
    // 之后会被替换成BatchedISendIRecvOp算子
    // Question: MakePlaceholderOp的各个参数是否都有必要
    auto needed_slice_instance = MakePlaceholderOp(owned_slice_instance->meta(), 
                                                   param->get_distributed_states(), 
                                                   OpMeta().set_device_group(group));
    param_slice->AddNeededSliceInst(device, needed_slice_instance);
    return needed_slice_instance;
  } 
  int32_t basic_slice_num = 0;
  TensorList merged_slices;
  auto it = state.find(dim);
  if (it != state.end()) {
    basic_slice_num = it->second * multiple[dim];
  }
  for (int32_t i = 0; i < multiple[dim]; ++i) {
    slice_num[dim] = basic_slice_num + i;
    slice_relative_num[dim] = i;
    Tensor merged_slice = MergeAllParamSlices(param, block, device, group, slice_num, slice_relative_num, state, multiple, dim + 1);
    merged_slices.push_back(std::move(merged_slice));
  }  
  auto concatenate_output = MakeConcatenateOp(std::move(merged_slices), dim, OpMeta().set_is_deduce_states(false));         
  auto& concatenate_op = concatenate_output->producer();
  // 其他device上生成的不需要map placement_group和placement
  if (hetu::impl::comm::GetLocalDevice() == device) { 
    concatenate_op->MapToParallelDevices(group);
    concatenate_op->Instantiate(device, kComputingStream);
  }  
  return concatenate_output;
}

// 对于一个param
// 进行全局的switch
// 每台机器会知道自己拥有这个param的哪些部分以及需要这个param的哪些部分
void SwitchExecGraph::SwitchParam(const DistributedStates& src_ds, const DeviceGroup& src_group,
                                   const DistributedStates& dst_ds, const DeviceGroup& dst_group,
                                   const Tensor& comm_input, const Tensor& after_param) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  std::vector<int32_t> block_shape;
  std::vector<int32_t> src_multiple;
  std::vector<int32_t> dst_multiple;
  // 获得最小粒度的块划分
  int32_t param_dims = comm_input->global_shape().size(); // size_t -> int32_t
  for (int32_t key = -2; key < param_dims; ++key) {
    auto src_dim = src_ds.states(key);
    auto dst_dim = dst_ds.states(key);
    if (key == -2) {
      HT_ASSERT(src_dim == 1 && dst_dim == 1) 
        << "parameter ds shouldn't have partial dim";
      continue;
    }
    if (key == -1) {
      continue;
    }
    auto max_dim = std::max(src_dim, dst_dim);
    HT_ASSERT(max_dim % src_dim == 0 && max_dim % dst_dim == 0)
      << "only support scaling by an integer";
    block_shape.push_back(max_dim); 
    src_multiple.push_back(max_dim / src_dim);
    dst_multiple.push_back(max_dim / dst_dim);
  }
  // 为当前param创建一个全局的、抽象的ParamBlock
  // 并为每个最小粒度的块划分创建一个抽象的ParamSlice
  // 其只需要知道ds的切分shape，而不需要绑定param的真实shape
  const TensorName& param_block_name = after_param->name();
  HT_LOG_DEBUG << local_device << ": make an abstract block for " << param_block_name
    << ", whose shape is " << block_shape;
  auto param_block_ptr = std::make_shared<ParamBlock>(param_block_name, block_shape, this);
  std::vector<int32_t> slice_num(param_dims, 0);
  CreateParamBlock(*param_block_ptr, slice_num, param_block_name, 0);
  _param_blocks.push_back(param_block_ptr);
  // 每个device作为发送端
  // 求出每个device拥有的小块儿并进行切分
  auto src_devices_size = src_group.num_devices();
  for(size_t i = 0; i < src_devices_size; ++i) {
    // 初始化发送端的mapping
    auto it = _send_mapping.find(src_group.get(i));
    if (it == _send_mapping.end()) {
      _send_mapping[src_group.get(i)] = std::make_pair(std::vector<Device>{}, std::vector<Tensor>{});
    }
    auto cur_state_index = src_ds.map_device_to_state_index(i);
    std::vector<int32_t> cur_slice_num(param_dims, 0);
    std::vector<int32_t> cur_slice_relative_num(param_dims, 0);
    // 进行具体的切分
    // 将ParamSliceInstance放入对应的ParamSlice
    HT_LOG_DEBUG << local_device << ": MakeAllParamSlices for tensor " << comm_input << " at device " << src_group.get(i)
      << ", cur_state_index = " << cur_state_index << " and src_multiple = " << src_multiple;
    MakeAllParamSlices(comm_input, *param_block_ptr, src_group.get(i), src_group, cur_slice_num, cur_slice_relative_num, 
                       cur_state_index, src_multiple, 0);
  }
  // 每个device作为接收端
  // 求出每个device需要的小块儿并进行合并
  auto dst_devices_size = dst_group.num_devices();
  for(size_t i = 0; i < dst_devices_size; ++i) {
    // 初始化发送端的mapping
    auto it = _recv_mapping.find(dst_group.get(i));
    if (it == _recv_mapping.end()) {
      _recv_mapping[dst_group.get(i)] = std::make_pair(std::vector<Device>{}, std::vector<Tensor>{});
    }
    auto cur_state_index = dst_ds.map_device_to_state_index(i);
    std::vector<int32_t> cur_slice_num(param_dims, 0);
    std::vector<int32_t> cur_slice_relative_num(param_dims, 0);
    // 进行具体的合并
    // 将新的ParamSliceInstance放入对应的ParamSlice
    // 会先用placeholder（之后再用BatchedISendIRecvOp进行替换）表征ParamSliceInstance
    // 返回的result即为新exec graph中最终合并后的param
    HT_LOG_DEBUG << local_device << ": MergeAllParamSlices for tensor " << after_param << " at device " << src_group.get(i)
      << ", cur_state_index = " << cur_state_index << " and dst_multiple = " << dst_multiple;
    auto result = MergeAllParamSlices(after_param, *param_block_ptr, dst_group.get(i), dst_group, cur_slice_num, cur_slice_relative_num, 
                                      cur_state_index, dst_multiple, 0);
    // 如果是local的result
    // 记录result以及其与after graph param的映射
    if (local_device == dst_group.get(i)) {
      _comm_results_mapping.insert(std::make_pair(result->id(), after_param));
      _comm_results.push_back(std::move(result));
    }
  }
}

void SwitchExecGraph::MakeCommGraph() {

  auto local_device = hetu::impl::comm::GetLocalDevice();
  HT_LOG_DEBUG << local_device << ": make a new comm graph begin...";

  auto& before_graph = _switch_graph_pair.first;
  auto& after_graph = _switch_graph_pair.second;
  auto& before_mapping = _define_graph->GetPlan(_switch_plan_pair.first).tensor_to_exec_tensor_mapping;
  auto& after_mapping = _define_graph->GetPlan(_switch_plan_pair.second).tensor_to_exec_tensor_mapping;
  _comm_graph = Graph::_make_new_graph<ExecutableGraph>(
    "comm_graph_between_" + before_graph->name() 
    + "_and_" + after_graph->name());

  Graph::push_graph_ctx(_comm_graph->id());
  
  std::unordered_set<Device> src_set;
  std::unordered_set<Device> dst_set;
  DataType dtype = DataType::UNDETERMINED;
  for (auto& define_param_ref : _define_graph_params) {
    auto& define_param = define_param_ref.get();
    // Test Case
    /*
    if ("wte_table" != define_param->name()) {
      continue;
    }
    */
    auto& param_global_shape = define_param->global_shape();
    if (dtype == DataType::UNDETERMINED) {
      dtype = define_param->dtype();
    } else {
      HT_ASSERT(dtype == define_param->dtype())
        << "we only support homogeneous dtype now, but there are two param dtypes: "
        << dtype << " and " << define_param->dtype();
    }
    auto define_param_id = define_param->id();
    auto before_it = before_mapping.find(define_param_id);
    bool is_before_active = before_it != before_mapping.end();
    auto after_it = after_mapping.find(define_param_id);
    bool is_after_active = after_it != after_mapping.end();
    // 分情况讨论
    HT_LOG_DEBUG << local_device << ": processing param " << define_param << " in switch from "
      << before_graph->name() << " to " << after_graph->name();
    if (!is_before_active && !is_after_active) {
      // 尽管在define and run graph中创建了某一param
      // 但在实际的exec graph中并没有进行使用
      // 例如lm_head_weight
      // 这种情况我们什么都不处理
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is not in both graphs";
    } else if (is_before_active && !is_after_active) {
      // TODO: save the param back to the cpu
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is only in the before graph";
      auto& before_param = before_it->second;
      HT_RUNTIME_ERROR << "NotImplementedError";
    } else if (!is_before_active && is_after_active) {
      // 为了保证正确性我们这里还是会从add init里再度对其赋值
      // 这里只对after的add init赋值而不会产生性能上的开销
      // 目的是为了防止在新的after graph中有新的topo而需要访问这一before graph中未使用的param
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is only in the after graph";
      auto& after_param = after_it->second;
      auto add_on_inits_it = _define_graph->_add_on_inits.find(define_param->id());
      if (add_on_inits_it != _define_graph->_add_on_inits.end()) {
        HT_LOG_DEBUG << local_device << ": param " << define_param << " in the after graph is reset";
        Graph::ResetVariableData(after_param, *add_on_inits_it->second);
      } else {
        // 另外一种情况是param不在_add_on_inits里
        // 即没有被修改过provided data
        // 这种情况不需要handle（exec graph的AllocVariableDataInner会自动帮忙处理）
        HT_LOG_DEBUG << local_device << ": param " << define_param << " in the after graph will be lazily initialized";
      }
    } else {
      // 这种情况才是我们核心要考虑的
      HT_LOG_DEBUG << local_device << ": param " << define_param << " is in both graphs";
      auto& before_param = before_it->second;
      auto& after_param = after_it->second;
      HT_ASSERT(before_param->global_shape() == after_param->global_shape())
        << "parameter shapes in the two switching exec graphs should be equal"
        << ", but find global shape of " << before_param << " in before graph is " << before_param->global_shape()
        << " and global shape of " << after_param << " in after graph is " << after_param->global_shape();
      const auto& src_ds = before_param->get_distributed_states();
      const auto& src_group = before_param->producer()->device_group();
      const auto& dst_ds = after_param->get_distributed_states();
      const auto& dst_group = after_param->producer()->device_group();
      // 将出现过的device都放入comm set中
      for (auto& device : src_group.devices()) {
        if (src_set.find(device) == src_set.end()) {
          src_set.insert(device);
        }
        // 用来之后BatchedIsendIrecv以及MPI同步的
        if (_comm_set.find(device) == _comm_set.end()) {
          _comm_set.insert(device);
        }
      }
      for (auto& device : dst_group.devices()) {
        if (dst_set.find(device) == dst_set.end()) {
          dst_set.insert(device);
        }
        // 用来之后BatchedIsendIrecv以及MPI同步的
        if (_comm_set.find(device) == _comm_set.end()) {
          _comm_set.insert(device);
        }
      }
      // 依据before_param生成通信图input的placeholder以及相应的feed_dict
      // 理论上也可以直接使用before_param作为input的tensor
      // 但这里还是希望尽量保证comm graph与before graph之间的隔离性
      auto comm_input = MakePlaceholderOp(before_param->meta(), 
                                          before_param->get_distributed_states(), 
                                          OpMeta().set_device_group(src_group).set_name(before_param->name() + "_comm_input"));
      if (src_group.contains(local_device)) {
        comm_input->producer()->MapToParallelDevices(src_group);
        comm_input->producer()->Instantiate(local_device, kComputingStream);
        auto comm_input_data_it = before_graph->_preserved_data.find(before_param->id());
        HT_ASSERT(comm_input_data_it != before_graph->_preserved_data.end())
          << "something wrong, the data to transfer in the before graph is not available";
        _comm_feed_dict_mapping.insert(std::make_pair(comm_input->id(), before_param));
        _comm_feed_dict.insert(std::make_pair(comm_input->id(), comm_input_data_it->second));
      }
      // 生成该param切分和合并的计算图
      // 并建立映射关系
      // 只是知道哪些device需要哪些slice
      // 哪些device拥有哪些slice
      // 不进行实际的算法决策
      HT_LOG_DEBUG << local_device << ": switch param from " << before_param << " to " << after_param
        << ", src group = " << src_group << " and dst_group = " << dst_group 
        << ", src ds states = " << src_ds.get_states() << " and dst states = " << dst_ds.get_states();
      SwitchParam(src_ds, src_group, dst_ds, dst_group, comm_input, after_param);
    }
  }

  // 从全局的ParamBlocks视角出发
  // 选择最优的通信方案
  // TODO: 更好的算法
  // 目前采用的是对于每一个ParamBlock的每一个ParamSlice，采用round robin的算法
  // 12.28，更新为greedy算法，选取p2p中通信次数最小的
  for (auto& param_block_ptr : _param_blocks) {
    param_block_ptr->ParamBlockComm(_send_mapping, _recv_mapping);
  }

  // _send_mapping和_recv_mapping此时已经获取到所有params的通信方案
  // 将中间的placeholder算子替换为具体的通信算子
  HT_LOG_DEBUG << local_device << ": make the crucial BatchedISendIRecvOp begin...";
  std::vector<Device> src_devices(src_set.begin(), src_set.end());
  std::vector<Device> dst_devices(dst_set.begin(), dst_set.end());
  std::vector<Device> comm_devices(_comm_set.begin(), _comm_set.end());
  // local_device is exclusive
  auto comm_device_group = DeviceGroup(comm_devices);
  if (!comm_device_group.contains(local_device)) {
    HT_LOG_DEBUG << local_device << ": no params can leverage hot switch";
    Graph::pop_graph_ctx();
    HT_LOG_DEBUG << local_device << ": make a new comm graph end...";
    return;
  }
  // local_device send to other devices
  std::vector<Device>& send_to_devices = _send_mapping[local_device].first;
  TensorList& send_tensors = _send_mapping[local_device].second;
  // local_device receive from other devices
  std::vector<Device>& recv_from_devices = _recv_mapping[local_device].first;
  HTShapeList recv_tensor_shapes;
  auto recv_len = _recv_mapping[local_device].second.size();
  for (size_t i = 0; i < recv_len; ++i) {
    recv_tensor_shapes.push_back(_recv_mapping[local_device].second[i]->shape());
  }
  // BatchedISendIRecv Part
  HT_LOG_DEBUG << local_device << ": will send " << send_tensors.size() << " tensor to device " 
    << send_to_devices << " and recv " << recv_len << " tensor from other devices"
    << ", the src devices = " << src_devices << " and comm devices = " << comm_devices;
  // 在通信前插入contiguous算子
  // profile时单独计时
  TensorList contiguous_send_tensors;
  contiguous_send_tensors.reserve(send_tensors.size());
  for (auto& send_tensor : send_tensors) {
    auto contiguous_send_tensor = MakeContiguousOp(send_tensor, 
                                                   OpMeta().set_is_deduce_states(false));
    auto& contiguous_op = contiguous_send_tensor->producer();
    HT_ASSERT(send_tensor->placement_group().contains(local_device))
      << "send tensor should already be instantiated locally";
    contiguous_op->MapToParallelDevices(send_tensor->placement_group());
    contiguous_op->Instantiate(local_device, kComputingStream);
    contiguous_send_tensors.push_back(std::move(contiguous_send_tensor));
  }
  auto result = MakeBatchedISendIRecvOp(contiguous_send_tensors, send_to_devices, 
                                        recv_tensor_shapes, recv_from_devices, 
                                        comm_devices, dtype, 
                                        OpMeta().set_is_deduce_states(false));
  auto& batched_isend_irecv_op = result->producer();
  batched_isend_irecv_op->MapToParallelDevices(comm_device_group);
  batched_isend_irecv_op->Instantiate(local_device, kP2PStream);
  TensorList recv_tensors = batched_isend_irecv_op->outputs();
  // we need to add dummy link for topo sort
  // 只有send没有recv
  // 要将这种情况的dummy link放到fetch中
  if (recv_from_devices.size() == 0) {
    HT_LOG_DEBUG << local_device << ": no recv from other devices";
    HT_ASSERT(result == batched_isend_irecv_op->out_dep_linker())
      << "something wrong, it should be the out_dep_linker";
    _dummy_links.push_back(result);
  } else {
    HT_LOG_DEBUG << local_device << ": recv from devices " << recv_from_devices;
  }
  HT_LOG_DEBUG << local_device << ": make the crucial " << result << " end..";

  // 将原先的placeholder替换为recv_tensor
  HT_ASSERT(recv_len == recv_tensors.size())
    << "something wrong with the recv len";
  for (size_t i = 0; i < recv_len; ++i) {
    auto& old_tensor = _recv_mapping[local_device].second[i];
    auto& new_tensor = recv_tensors[i];
    HT_ASSERT(old_tensor->num_consumers() == 1)
      << "the slice instance should only used once (by a single concatenate op)";
    auto& consumer = old_tensor->consumer(0);
    for (size_t j = 0; j < consumer->num_inputs(); ++j) {
      if (consumer->input(j)->id() == old_tensor->id()) {
        Graph::ReplaceInput(consumer, j, new_tensor);
      }
    }
  }

  Graph::pop_graph_ctx();
  HT_LOG_DEBUG << local_device << ": make a new comm graph end...";
}

// context switch
// 将before graph中的所有params以尽量高效的方式
// 重新分配到after graph中
void SwitchExecGraph::SwitchParams() {

  // utils
  auto local_device = hetu::impl::comm::GetLocalDevice();
  auto is_feed_dict_op = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return _comm_feed_dict.find(tensor->id()) != _comm_feed_dict.end();
    });
  };
  // 如果有cache好的_comm_graph
  // 那么直接使用即可
  // 否则需要重新建立
  if (_comm_graph != nullptr) {
    // 只需要重新设置_comm_feed_dict即可
    // 从_preserve_data中获取before graph的params的数据
    for (const auto& kv : _comm_feed_dict_mapping) {
      auto comm_feed_dict_it = _comm_feed_dict.find(kv.first);
      HT_ASSERT(comm_feed_dict_it != _comm_feed_dict.end())
        << "_comm_feed_dict_mapping is wrong";
      auto comm_input_data_it = _switch_graph_pair.first->_preserved_data.find(kv.second->id());
      HT_ASSERT(comm_input_data_it != _switch_graph_pair.first->_preserved_data.end())
      << "something wrong, the data to transfer in the before graph is not available";
      // 给feed_dict赋上NDArray
      comm_feed_dict_it->second = comm_input_data_it->second;
    }
  } else {
    TIK(switch_params_making);
    HT_ASSERT(_comm_results.empty())
      << "no comm result should exist";
    // *建图*
    MakeCommGraph();
    // 计算topo
    HT_LOG_DEBUG << local_device << ": the mutual params len is " << _param_blocks.size()
      << " and the local recv params len is " << _comm_results.size();
    TensorList fetches(_comm_results);
    fetches.insert(fetches.end(), _dummy_links.begin(), _dummy_links.end());
    OpRefList topo = Graph::TopoSort(fetches, -1, is_feed_dict_op);
    // HT_LOG_DEBUG << local_device << ": global topo of the comm graph is " << topo;
    // 本地topo
    auto get_local_topo = [&](OpRefList& topo, OpRefList& local_topo) {
      for (auto& op_ref : topo) {
        HT_ASSERT(op_ref.get()->placement().type() != DeviceType::UNDETERMINED)
          << "op " << op_ref.get() << " in comm graph is not instantiated";
        if (op_ref.get()->placement() == local_device) {
          local_topo.push_back(op_ref);
        }
      }
    };
    get_local_topo(topo, _comm_topo);
    HT_LOG_DEBUG << local_device << ": local topo of the comm graph is " << _comm_topo;
    // 计算运行时shape
    // 该图中只存在placeholder、split、batchedisendirecv和concatenate
    // 不需要symbolic方法（甚至不需要DoInferShape）
    // 直接用tensor的shape即可
    // Question: 是否正确？
    for (auto& op_ref : _comm_topo) {
      auto& op = op_ref.get();
      for (const auto& output : op->outputs()) {
        _comm_shape_plan[output->id()] = output->shape();
      }
    }
    if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
      TOK(switch_params_making);
      HT_LOG_INFO << local_device << ": switch params making graph & plan time = " << COST_MSEC(switch_params_making) << " ms";
    }
  }

  // profile时需要先都同步好了
  if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
    // stream同步
    SynchronizeAllStreams(local_device);
    // rank同步
    // 这里对全部rank进行同步
    /*
    if (!_comm_set.empty()) {
      std::vector<Device> mpi_devices(_comm_set.begin(), _comm_set.end());
      DeviceGroup mpi_device_group{mpi_devices};
      auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreate(hetu::impl::comm::DeviceGroupToWorldRanks(mpi_device_group));
      mpi_group->Barrier(true);
      HT_LOG_DEBUG << local_device << ": params switch comm set = " << mpi_device_group;
    }
    */
    auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreateWorldwide();
    mpi_group->Barrier(true);
    if (_profile_level <= SWITCH_PROFILE_LEVEL::NVLINK) {
      ProfileNvlinkStart();
    }
    mpi_group->Barrier(true);
  }
  
  // 启动！
  TIK(switch_params_running); // 开始计时
  Tensor2NDArrayMap tensor2data;
  Tensor2IntMap tensor2degrees;
  RuntimeContext runtime_ctx(_comm_topo.size(), _comm_shape_plan);
  // 计算各个tensor的度
  for (auto& op_ref : _comm_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      tensor2degrees[input->id()]++;
    }
  }
  for (auto& op_ref : _comm_topo) {
    auto& op = op_ref.get();
    HT_LOG_DEBUG << local_device << ": handling op " << op << " in comm graph";
    if (is_feed_dict_op(op)) {
      // 对于feed_dict只需要简单地设置data即可
      // 可以保证这里全都是只有一个输出的placeholder
      HT_ASSERT(is_placeholder_op(op))
        << "feed dict op must be a placeholder in the comm graph";
      tensor2data[op->output(0)->id()] = _comm_feed_dict[op->output(0)->id()];
      continue;
    }
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      auto it = tensor2data.find(input->id());
      HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
        << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << input;
      auto& data = it->second;
      HT_ASSERT(data->device() == input->placement() && data->dtype() == input->dtype())
        << local_device << ": Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "input " << input << " placement/dtype is wrong";
      input_vals.push_back(tensor2data[input->id()]);
      // free memory after op async compute complete
      if ((--tensor2degrees[input->id()]) == 0) {
        tensor2data.erase(input->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    // Note: The usage should be marked inside kernels, 
    // but we still mark here in case we forget to do so in some kernels. 
    NDArray::MarkUsedBy(input_vals, op->instantiation_ctx().stream());
    NDArray::MarkUsedBy(output_vals, op->instantiation_ctx().stream());
    // 记录输出值
    for (size_t i = 0; i < op->num_outputs(); ++i) {
      tensor2data[op->output(i)->id()] = output_vals[i];
    }
  }

  // 将结果赋值给after graph
  for (const auto& kv : _comm_results_mapping) {
    auto _comm_results_it = tensor2data.find(kv.first);
    HT_ASSERT(_comm_results_it != tensor2data.end())
      << "something wrong, can't find the result from the tensor2data mapping";
    HT_LOG_DEBUG << local_device << ": comm result sum of " << kv.second << " is " << NDArray::sum(_comm_results_it->second);
    // 给新图的_preserved_data赋上NDArray
    _switch_graph_pair.second->_preserved_data[kv.second->id()] = _comm_results_it->second;
  }

  // 将before graph中保留的数据全部清除
  // TODO: 最坏情况需要1.5倍的显存开销，后续需要分bucket进行发送并清除
  _switch_graph_pair.first->_preserved_data.clear();

  if (_profile_level < SWITCH_PROFILE_LEVEL::INFO) {
    // stream同步
    TensorList fetches(_comm_results);
    fetches.insert(fetches.end(), _dummy_links.begin(), _dummy_links.end());
    for (const auto fetch : fetches) {
      fetch->producer()->Sync();
    }  
    TOK(switch_params_running); // 结束计时
    // rank同步（不同rank耗时不一样，因此放在TOK之后）
    // 这里对全部rank进行同步
    /*
    if (!_comm_set.empty()) {
      std::vector<Device> mpi_devices(_comm_set.begin(), _comm_set.end());
      DeviceGroup mpi_device_group{mpi_devices};
      auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreate(hetu::impl::comm::DeviceGroupToWorldRanks(mpi_device_group));
      mpi_group->Barrier(true);
    }
    */
    auto& mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreateWorldwide();
    mpi_group->Barrier(true);
    HT_LOG_INFO << local_device << ": switch params running time = " << COST_MSEC(switch_params_running) << " ms";
    if (_profile_level <= SWITCH_PROFILE_LEVEL::NVLINK) {
      ProfileNvlinkEnd();
    }
    ProfileRunningDetails();
  }
  
  HT_LOG_DEBUG << local_device << ": params switch from " << _switch_graph_pair.first->name()
   << " to " << _switch_graph_pair.second->name() << " is done";
  // HT_RUNTIME_ERROR << local_device << ": breakpoint";
}

// profile details
void SwitchExecGraph::ProfileRunningDetails() {

  HT_ASSERT(_comm_graph != nullptr)
    << "Profiler can only used after comm graph was built";
  auto local_device = hetu::impl::comm::GetLocalDevice();
  size_t slice_num = 0, concat_num = 0, contiguous_num = 0, comm_num = 0;
  double slice_time = 0, concat_time = 0, contiguous_time = 0, comm_time = 0;
  
  // op execute time
  for (auto& op_ref : _comm_topo) {
    auto& op = op_ref.get();
    // Note: only one micro batch
    if (is_placeholder_op(op)) {
      continue;
    } else if (is_slice_op(op)) {
      slice_time += op->TimeCost(0) * 1.0 / 1e6;
      slice_num += 1;
    } else if (is_concat_op(op)) {
      concat_time += op->TimeCost(0) * 1.0 / 1e6;
      concat_num += 1;
    } else if (is_contiguous_op(op)) {
      contiguous_time += op->TimeCost(0) * 1.0 / 1e6;
      contiguous_num += 1;
    } else if (is_batched_isend_irecv_op(op)) {
      comm_time += op->TimeCost(0) * 1.0 / 1e6;
      comm_num += 1;
    } else {
      HT_RUNTIME_ERROR << local_device << ": op " << op 
        << " shouldn't exit in the comm graph";
    }
  }

  // comm detailed info
  std::vector<Device>& send_to_devices = _send_mapping[local_device].first;
  TensorList& send_tensors = _send_mapping[local_device].second;
  std::unordered_map<Device, std::vector<std::string>> send_info_mapping;
  std::ostringstream send_info_output;
  auto send_len = send_to_devices.size();
  HT_ASSERT(send_tensors.size() == send_len)
    << "something wrong with the size";
  for (size_t i = 0; i < send_len; ++i) {
    auto it = _info_mapping.find(send_tensors[i]->id());
    HT_ASSERT(it != _info_mapping.end())
      << "send tensor info is not existed";
    send_info_mapping[send_to_devices[i]].push_back(it->second);
  }
  for (const auto& kv : send_info_mapping) {
    send_info_output << "send " << kv.second.size()
      << " tensor to " << kv.first;
    if (_profile_level == SWITCH_PROFILE_LEVEL::TRACE) {
      for (const auto& send_info : kv.second) {
        send_info_output << ", " << send_info;
      }
    }
    send_info_output << std::endl;
  }

  HT_LOG_INFO << local_device << ": switch params running details: " << std::endl
    << "*********************************************" << std::endl
    << "slice num = " << slice_num << ", time = " << slice_time << " ms" << std::endl
    << "concat num = " << concat_num << ", time = " << concat_time << " ms" << std::endl
    << "contiguous num = " << contiguous_num << ", time = " << contiguous_time << " ms" << std::endl
    << "comm num = " << comm_num << ", time = " << comm_time << " ms" << std::endl
    << "*********************************************" << std::endl
    << send_info_output.str()
    << "*********************************************";
}

void SwitchExecGraph::ProfileNvlinkStart() {

  // 只需要一个机器profile即可
  if (hetu::impl::comm::GetWorldRank() != 0) {
    return;
  }
  HT_LOG_INFO << "********* Profile NVLink Start *********";

  // 初始化NVML库
  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to initialize NVML: " << nvmlErrorString(result);
    return;
  }

  // 获取GPU数量
  result = nvmlDeviceGetCount(&_device_count);
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to query device count: " << nvmlErrorString(result);
    return;
  }
  _nvlink_counts.reserve(_device_count);
  _nvlink_txs.reserve(_device_count);
  _nvlink_rxs.reserve(_device_count);

  for (unsigned int i = 0; i < _device_count; ++i) {
    // Initialization
    _nvlink_counts.emplace_back(0);
    _nvlink_txs.emplace_back();
    _nvlink_rxs.emplace_back();

    // Get current device
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
      return;
    }

    // Check the NVLink status for each possible link
    for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
      nvmlEnableState_t is_active;
      result = nvmlDeviceGetNvLinkState(device, link, &is_active);
      if (NVML_SUCCESS == result && is_active == NVML_FEATURE_ENABLED) {
        _nvlink_counts[i]++;
      }
    }
    HT_LOG_INFO << "GPU " << i << " has " << _nvlink_counts[i] << " NVLink connections active";
    if (_nvlink_counts[i] == 0) {
      continue;
    }
    _nvlink_txs.reserve(_nvlink_counts[i]);
    _nvlink_rxs.reserve(_nvlink_counts[i]);

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 记录执行nccl通信代码片段前的NVLink Raw Tx和Raw Rx
    result = nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data());
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get utilization control: " << nvmlErrorString(result);
      return;
    }
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      _nvlink_txs[i].emplace_back(field_values[2 * link].value.ullVal);
      _nvlink_rxs[i].emplace_back(field_values[2 * link + 1].value.ullVal);
    }
  }
}


void SwitchExecGraph::ProfileNvlinkEnd() {

  // 只需要一个机器profile即可
  // 如果没有NVLink则不再profile
  if (hetu::impl::comm::GetWorldRank() != 0) {
    return;
  }

  nvmlReturn_t result;
  for (unsigned int i = 0; i < _device_count; ++i) {
    if (_nvlink_counts[i] == 0) {
      continue;
    }
    // Get current device
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
      return;
    }

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 获取执行nccl通信代码片段后的NVLink Raw Tx和Raw Rx
    result = nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data());
    if (result != NVML_SUCCESS) {
      HT_RUNTIME_ERROR << "Failed to get utilization control: " << nvmlErrorString(result);
      return;
    }
    for (unsigned int link = 0; link < _nvlink_counts[i]; ++link) {
      _nvlink_txs[i][link] = field_values[2 * link].value.ullVal - _nvlink_txs[i][link];
      _nvlink_rxs[i][link] = field_values[2 * link + 1].value.ullVal - _nvlink_rxs[i][link];
      // 打印NVLink Raw Tx和Raw Rx的变化量
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Tx Delta: " << _nvlink_txs[i][link] << " KiB";
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Rx Delta: " << _nvlink_rxs[i][link] << " KiB";
    }
  }

  // 清理NVML资源
  result = nvmlShutdown();
  if (result != NVML_SUCCESS) {
    HT_RUNTIME_ERROR << "Failed to shutdown NVML: " << nvmlErrorString(result);
    return;
  }
   
  HT_LOG_INFO << "********* Profile NVLink End *********";
}

} // namespace graph
} // namespace hetu
