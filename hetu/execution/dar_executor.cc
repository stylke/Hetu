#include "hetu/execution/dar_executor.h"
#include "hetu/execution/device_placer.h"
#include "hetu/execution/run_metadata.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/dataloader.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/autograd/ops/Split.h"
#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/Concatenate.h"
#include "hetu/autograd/distributed_states.h"
#include <queue>

namespace hetu {
namespace execution {

using namespace hetu::impl::comm;
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
    // // 1. 推导每个tensor的states/order
    // DeduceDistributedStates(_exec_ctx->_global_topo_order);
    // 2. 获取local topo
    _exec_ctx->_local_topo_order =
      get_local_nodes(_exec_ctx->_global_topo_order, _exec_ctx->_local_device);
    HT_LOG_DEBUG << "Local topo: " << _exec_ctx->_local_topo_order;
    // 3. 将local_topo中的comm_op替换为具体的通信op
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": Substitute comm ops begin!";
    SubstituteCommOp(_exec_ctx->_local_topo_order);
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": Substitute comm ops end!";

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
// 推导所有tensor的distributed_states, 这个后续再写
void DARExecutor::DeduceDistributedStates(const OpList& topo_order) {
  for (auto& op : topo_order) {
    if (is_comm_op(op)) {
      CommOp& op = reinterpret_cast<CommOp&>(op);
      auto src_distributed_states = op->input(0)->get_distributed_states();
      auto dst_distributed_states = op->get_dst_distributed_states();
      HT_ASSERT(src_distributed_states.is_valid() && dst_distributed_states.is_valid() && 
                src_distributed_states.get_device_num() == dst_distributed_states.get_device_num())
                << "cannot convert src distributed states to unpaired dst distributed states!";
      op->output(0)->set_distributed_states(dst_distributed_states);
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
  for (auto op : local_topo_order) {
    if (is_comm_op(op)) {
      HT_LOG_DEBUG << _exec_ctx->local_device() << "==============> substitute comm_op begin: " << op << "...";
      auto& comm_op = reinterpret_cast<CommOp&>(op);
      uint64_t comm_type = comm_op->get_comm_type();
      Tensor result;
      // TensorList extra_in_deps;
      if (comm_type == ALL_REDUCE_OP) {
        DeviceGroup comm_group = comm_op->get_devices_by_dim(-2); // do allreduce among comm_group
        AllReduceOp all_reduce_op(
          comm_op->input(0), comm_group, // comm_group is a subset of comm_op's placement_group
          OpMeta().set_device_group(comm_op->placement_group()).set_name(comm_op->input(0)->name() + "_AllReduce"));
        all_reduce_op->MapToParallelDevices(comm_op->placement_group());
        all_reduce_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
        result = all_reduce_op->output(0);
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": substitute comm_op to all_reduce_op: " << comm_group;        
      } else if (comm_type == ALL_GATHER_OP) {
        DeviceGroup comm_group = comm_op->get_devices_by_dim(0);
        AllGatherOp all_gather_op(
          comm_op->input(0), comm_group,
          OpMeta().set_device_group(comm_op->placement_group()).set_name(comm_op->input(0)->name() + "_AllGather"));
        all_gather_op->MapToParallelDevices(comm_op->placement_group());
        all_gather_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
        result = all_gather_op->output(0);
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": substitute comm_op to all_gather_op: " << comm_group;
      } else if (comm_type == REDUCE_SCATTER_OP) {
        DeviceGroup comm_group = comm_op->get_devices_by_dim(-2);
        ReduceScatterOp reduce_scatter_op(
          comm_op->input(0), comm_group,
          OpMeta().set_device_group(comm_op->placement_group()).set_name(comm_op->input(0)->name() + "_ReduceScatter"));
        reduce_scatter_op->MapToParallelDevices(comm_op->placement_group());
        reduce_scatter_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
        result = reduce_scatter_op->output(0);
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": substitute comm_op to reduce_scatter_op: " << comm_group;
      } else if (comm_type == P2P_OP) {
        // 分为两部分: 1. 从local_device发送数据给需要的device 2. 从目标devices中接收需要的数据到local_device
        int32_t local_device_index = comm_op->placement_group().get_index(comm_op->placement());
        TensorList send_datas_local;
        std::vector<int32_t> dsts_local;
        HTShapeList recv_shapes_local;
        std::vector<int32_t> srcs_local;
        Tensor self_send_data;
        for (int32_t used_device_index = 0; used_device_index < comm_op->placement_group().num_devices(); used_device_index++) {     
          HT_LOG_DEBUG << _exec_ctx->local_device() << ": cross send begin!";
          int32_t device_index = 0;
          TensorList send_datas;
          std::vector<int32_t> dsts;
          // 执行其他device的cross send是为了获得recv_shapes信息
          CrossSend({}, {}, 0, false, device_index, comm_op, send_datas, dsts, used_device_index);
          HT_ASSERT(device_index == _exec_ctx->device_group().num_devices()) << "cross send error!";
          HT_LOG_DEBUG << _exec_ctx->local_device() << ": cross send end!";
          // for batch send/recv
          for (int i = 0; i < dsts.size(); i++) {
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

        std::vector<Device> dst_devices(dsts_local.size());
        std::vector<Device> src_devices(srcs_local.size());
        std::transform(dsts_local.begin(), dsts_local.end(), dst_devices.begin(), [&](int32_t index) { return comm_op->placement_group().get(index); });
        std::transform(srcs_local.begin(), srcs_local.end(), src_devices.begin(), [&](int32_t index) { return comm_op->placement_group().get(index); });
        BatchedISendIRecvOp batched_isend_irecv_op(send_datas_local, dst_devices, recv_shapes_local, src_devices, send_datas_local[0]->dtype(), OpMeta().set_name("BatchedISendIRecvOp_for_" + comm_op->name()));
        batched_isend_irecv_op->MapToParallelDevices(comm_op->placement_group());
        batched_isend_irecv_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
        TensorList recv_datas_local = batched_isend_irecv_op->outputs();

        HT_LOG_DEBUG << _exec_ctx->local_device() << ": cross receive begin!";
        int32_t device_index = 0;
        // 由于已经通过batch send/recv拿到了recv_datas, 只需要执行local device的cross receive即可
        result = CrossReceive(0, device_index, comm_op, recv_datas_local, srcs_local, self_send_data, local_device_index);
        HT_ASSERT(device_index == _exec_ctx->device_group().num_devices()) << "cross receive error!";
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": cross receive end!";    
      }
      
      result->set_distributed_states(comm_op->get_dst_distributed_states()); // 替换comm_op后得到的result tensor需要设置为dst_distributed_states

      // 找到所有消费tensor的op, 替换对应的input: 从占位符comm_op的output(0)替换为真正通信算子得到的result
      // size_t是unsigned的, 永远>=0; 此外comm_op->output(0)->num_consumers()是size_t, 如果是0, 则-1是一个无限大的数, 这里要闭坑
      for (int i = comm_op->output(0)->num_consumers() - 1; i >= 0; i--) { // size_t是一个坑, i从小到大遍历是一个坑
        // if (!extra_in_deps.empty()) { // 其他devices替换comm_op所产生的op需要被添加依赖才能在topo中被遍历到
        //   comm_op->output(0)->consumer(i)->AddInDeps(extra_in_deps);
        // }
        for (int j = 0; j < comm_op->output(0)->consumer(i)->num_inputs(); j++) {
          if (comm_op->output(0)->consumer(i)->input(j)->id() == comm_op->output(0)->id()) {
            comm_op->output(0)->consumer(i)->ReplaceInput(j, result); // op->ReplaceInput()原先是protected, 这里暂时挪到public来              
          }
        }
      }
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": replace end!";
      HT_LOG_DEBUG << _exec_ctx->local_device() << "==============> substitute comm_op end: " << op << "...";
    }
  }
}

// 给替换某个comm_op所生成的send/recv ops之间排一个拓扑序, 用于后续topo sort用, 避免死锁
void DARExecutor::GenerateSendRecvTopo(
  CommOp& comm_op, std::vector<P2PSendOp>& send_ops, 
  std::vector<P2PRecvOp>& recv_ops, 
  std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<int32_t>>>& send_recv_list) {
  int32_t local_device_index = comm_op->placement_group().get_index(comm_op->placement());
  std::unordered_map<int32_t, std::unordered_map<int32_t, std::unordered_map<int32_t, bool>>> is_visited;
  int32_t total_num = 0;
  for (int32_t index = 0; index < comm_op->placement_group().num_devices(); index++) {
    for (int32_t i = 0; i < send_recv_list[index][0].size(); i++) {
      int32_t dst = send_recv_list[index][0][i];
      if (dst != index) {    
        is_visited[0][index][dst] = false;
        total_num++;
      } else {
        is_visited[0][index][dst] = true;
      }
      if (local_device_index == 0) HT_LOG_DEBUG << index << " send to " << dst;
    }
    for (int32_t i = 0; i < send_recv_list[index][1].size(); i++) {
      int32_t src = send_recv_list[index][1][i];
      if (src != index) {  
        is_visited[1][index][src] = false;
        total_num++;
      } else {
        is_visited[1][index][src] = true;
      }
      if (local_device_index == 0) HT_LOG_DEBUG << index << " recv from " << src;
    }
  }

  std::vector<std::tuple<int32_t, int32_t, int32_t>> send_recv_order;
  int32_t next = -1;
  bool find_next = false;
  while (send_recv_order.size() < total_num) {
    if (!find_next) {
      for (int32_t index = 0; index < comm_op->placement_group().num_devices(); index++) {
        for (int32_t i = 0; i < send_recv_list[index][0].size(); i++) {
          int32_t dst = send_recv_list[index][0][i];
          if (!is_visited[0][index][dst]) {
            send_recv_order.push_back({0, index, dst});
            send_recv_order.push_back({1, dst, index});
            is_visited[0][index][dst] = true;
            is_visited[1][dst][index] = true;
            next = dst;
            find_next = true;     
            break;
          }
        }
        if (find_next) break;
      }
    }

    HT_ASSERT(find_next || !find_next && send_recv_order.size() < total_num) 
              << "send_recv_order: generate send recv order error!";

    find_next = false;
    for (int32_t i = 0; i < send_recv_list[next][0].size(); i++) {
      int32_t dst = send_recv_list[next][0][i];
      if (!is_visited[0][next][dst]) {
        send_recv_order.push_back({0, next, dst});
        send_recv_order.push_back({1, dst, next});
        is_visited[0][next][dst] = true;
        is_visited[1][dst][next] = true;
        auto old_next = next;
        next = dst;
        find_next = true;     
        break;
      }
    }
  }

  // if (local_device_index == 0) {
  //   for (int32_t i = 0; i < send_recv_order.size(); i++) {
  //     HT_LOG_DEBUG << "send_recv_order[" << i << "]: " << std::get<0>(send_recv_order[i]) << ", " 
  //                  << std::get<1>(send_recv_order[i]) << ", " << std::get<2>(send_recv_order[i]);
  //   }
  // }

  OpList send_recv_topo;
  for (auto& send_recv : send_recv_order) {
    if (std::get<0>(send_recv) == 0) {
      for (auto& send_op : send_ops) {
        auto src = std::get<1>(send_recv);
        auto dst = std::get<2>(send_recv);
        if (send_op->index_in_group() == src && send_op->dst_device_index() == dst) {
          send_recv_topo.push_back(send_op);
          break;
        }
      }
    }
    if (std::get<0>(send_recv) == 1) {
      for (auto& recv_op : recv_ops) {
        auto src = std::get<2>(send_recv);
        auto dst = std::get<1>(send_recv);
        if (recv_op->src_device_index() == src && recv_op->index_in_group() == dst) {
          send_recv_topo.push_back(recv_op);
          break;
        }
      }
    }
  }
  HT_ASSERT(send_recv_topo.size() == send_recv_order.size()) << "send_recv_topo size " << send_recv_topo.size() 
  << " must be equal to send_recv_order size " << send_recv_order.size() << "; send_recv_topo: " << send_recv_topo;

  // 在进行topo sort的时候, 替换comm_op的p2p send/recv op就必须要按照local_send_recv_topo的顺序来, 以便避免死锁
  for (auto& send_op : send_ops) {
    send_op->set_send_recv_topo(send_recv_topo);
  }
  for (auto& recv_op : recv_ops) {
    recv_op->set_send_recv_topo(send_recv_topo);
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": " << comm_op->name() << " send_recv_topo: " << send_recv_topo;  
}

// cross send的cur_state_index记录的是local device在prev states/order下分到了哪些数据, 通过device_index来记录local device需要发送的目标devices,
// 并将split或原数据发送给它们; cross receive的cur_state_index记录的是local device在target states/order下需要的是哪些数据, 通过device_index记录
// 这些所需数据来源的device, 将它们传来的数据进行sum/concatenate作为distributed tensor在local device转换后的part result tensor
Tensor DARExecutor::CrossReceive(
  int32_t depth, int32_t& device_index, CommOp& comm_op, 
  TensorList& recv_datas, std::vector<int32_t>& srcs,
  Tensor& self_send_data, int32_t& used_device_index) {
  auto prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  auto prev_order = prev_distributed_states.get_order();
  auto loop_sizes = prev_distributed_states.get_loop_sizes();

  auto target_distributed_states = comm_op->get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  HT_ASSERT(comm_op->placement() == _exec_ctx->local_device()) 
          << "Corss Receive: comm_op's placement " << comm_op->placement() << " != " << "local device " << _exec_ctx->local_device();
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
      HT_ASSERT(self_send_data->is_tensor()) << "Cross Receive: self_send_data must be a valid tensor!";
      result = self_send_data;
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": recv from device " << device_index << " don't need irecv";
    } else {
      for (int i = 0; i < srcs.size(); i++) {
        if (srcs[i] == device_index) {
          result = recv_datas[i];
          break;
        }
      }
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": recv from device " << device_index;      
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
      SumOp sum_op(part_result_list);
      if (used_device_index == local_device_index) {
        sum_op->MapToParallelDevices(comm_op->placement_group());
        sum_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
      }
      result = sum_op->output(0);    
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
      auto pre_st = prev_distributed_states.get_states()[cur_dim];
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
          ConcatenateOp concatenate_op(part_result_list, cur_dim);
          if (used_device_index == local_device_index) {
            concatenate_op->MapToParallelDevices(comm_op->placement_group());
            concatenate_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
          }
          result = concatenate_op->output(0);
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

void DARExecutor::CrossSend(
  std::unordered_map<int32_t, int32_t> split_cur_state, 
  std::unordered_map<int32_t, int32_t> split_target_state,
  int32_t depth, bool need_split, int32_t& device_index, 
  CommOp& comm_op, TensorList& send_datas, std::vector<int32_t>& dsts,
  int32_t& used_device_index) {
  // basic info
  auto prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  HT_ASSERT(comm_op->placement() == _exec_ctx->local_device()) 
          << "Corss Send: comm_op's placement " << comm_op->placement() << " != " << "local device " << _exec_ctx->local_device();
  auto local_device_index =comm_op->placement_group().get_index(comm_op->placement());  
  auto cur_state_index = prev_distributed_states.map_device_to_state_index(used_device_index); // 根据指定的device index和order确定该device拥有的是tensor的哪部分数据

  auto target_distributed_states = comm_op->get_dst_distributed_states();
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
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << " don't need to send to other devices!";
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
      // HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": new SplitOp: input(comm_op->input(0)).shape = " << comm_op->input(0)->shape();
      SplitOp split_op(comm_op->input(0), keys, indices, splits);
      if (used_device_index == local_device_index) { // 其他device上生成的用于替换comm_op不需要map placement_group和placement
        split_op->MapToParallelDevices(comm_op->placement_group());
        split_op->PlaceToLocalDevice(_exec_ctx->local_device(), kComputingStream);
      }
      send_part = split_op->output(0);
    } else {
      // 如果不需要split, 则发送整个tensor
      send_part = comm_op->input(0);
    }
    if (device_index == used_device_index) {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": send to device " << device_index << " don't need isend";
    } else {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": send to device " << device_index;
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
          HT_LOG_DEBUG << _exec_ctx->local_device() << ": device " << used_device_index << ": don't need to send to other devices!";
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
      auto tar_st = target_distributed_states.get_states()[cur_dim];
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
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": global topo: " << topo_order;
    topo_order = get_local_nodes(topo_order, _exec_ctx->_local_device, false);
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": local topo: " << topo_order;
    
    OpList local_fw_topo, local_bw_topo;
    if (_exec_ctx->is_pipeline_parallel()) {
      auto parts = disentangle_forward_and_backward_nodes(
        topo_order, _exec_ctx->_losses, true);
      local_fw_topo =
        get_local_nodes(std::get<0>(parts), _exec_ctx->_local_device, false);
      local_bw_topo =
        get_local_nodes(std::get<1>(parts), _exec_ctx->_local_device, false);
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

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": sub executor run begin!";

  // compute
  for (auto& op : _computing_ops) {
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": Executing op \"" << op->name() << "\"...";
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT(it != edge2arr.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      input_vals.push_back(it->second);
      // if (is_all_reduce_op(op)) {
      //   HT_LOG_DEBUG << _exec_ctx->local_device() << ": " << op->name() << ": in_edge: " << in_edge << ", val: " << it->second;
      // }
      if ((--edge2degrees[in_edge->id()]) == 0 &&
          fetch_ids.find(in_edge->id()) == fetch_ids.end()) {
        edge2arr.erase(in_edge->id());
      }
    }
    // HT_LOG_DEBUG << _exec_ctx->local_device() << ": " << op->name() << ": inputs: " << input_vals;
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    // HT_LOG_DEBUG << _exec_ctx->local_device() << ": " << op->name() << ": outputs: " << output_vals;
    HT_LOG_DEBUG << _exec_ctx->local_device() << ": " << op->name() << ": inputs: " << input_vals << "; outputs: " << output_vals; 
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      if (edge2degrees[out_edge->id()] > 0 ||
          fetch_ids.find(out_edge->id()) != fetch_ids.end()) {
        edge2arr[out_edge->id()] = output_vals[i];
      }
    }
  }

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": sub executor run end!";

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
  HT_LOG_INFO << _exec_ctx->local_device() << ": fw_topo_order: " << _fw_topo_order << "; bw_topo_order: " << _bw_topo_order; 
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
    HT_LOG_INFO << _exec_ctx->local_device() << ": exec op " << op;
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
