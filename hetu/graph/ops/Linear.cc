#include "hetu/graph/ops/Linear.h"
#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/profiler.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include <vector>

namespace hetu {
namespace graph {

void LinearOpImpl::DoCompute(Operator& op,const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  if (inputs.size() == 2)
    NDArray::linear(inputs.at(0), inputs.at(1), NDArray(), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
  else if (inputs.size() == 3)
    NDArray::linear(inputs.at(0), inputs.at(1), inputs.at(2), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList LinearOpImpl::DoGradient(Operator& op,const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = op->grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = Linear(a, b)
    // grad_a = Linear(grad_c, b^T), grad_b = Linear(a^T, grad_c)
    grad_a = op->requires_grad(0) ? MakeLinearOp(grad_c, b, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(a, grad_c, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = Linear(a^T, b)
    // grad_a = Linear(b, grad_c^T), grad_b = Linear(a, grad_c)
    grad_a = op->requires_grad(0) ? MakeLinearOp(b, grad_c, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(a, grad_c, false, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = Linear(a, b^T)
    // grad_a = Linear(grad_c, b), grad_b = Linear(grad_c^T, a)
    grad_a = op->requires_grad(0) ? MakeLinearOp(grad_c, b, false, false, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(grad_c, a, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else {
    // case 4: c = Linear(a^T, b^T)
    // grad_a = Linear(b^T, grad_c^T), grad_b = Linear(grad_c^T, a^T)
    grad_a = op->requires_grad(0) ? MakeLinearOp(b, grad_c, true, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(grad_c, a, true, true, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  }
  if (op->num_inputs() == 2) {
    return {grad_a, grad_b};
  } else if (op->num_inputs() == 3) {
    Tensor grad_bias = op->requires_grad(2) ? MakeReduceOp(grad_outputs.at(0), ReductionType::SUM, {0}, {false},
                                           g_op_meta.set_name(op->grad_name(2)))
                                         : Tensor();
    return {grad_a, grad_b, grad_bias};
  }
}

HTShapeList LinearOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  HT_ASSERT(a.size() == 2 && b.size() == 2 &&
            a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
    << "Invalid input shapes for " << op << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
}

DistributedStates deduce_states(const DistributedStates& ds_a, const DistributedStates& ds_b, bool trans_a, bool trans_b) {
  int32_t device_num = ds_a.get_device_num();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid()
            && ds_a.get_device_num() == ds_b.get_device_num())
            << "distributed states for Tensor a & Tensor b should be valid!";  
  // l,r to result states map  
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, 1}, {0, 0}, {1, -2}}, // no trans
    {{-1, 1}, {1, 0}, {0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, 0}, {0, -2}, {1, 1}}, // no trans
    {{-1, 0}, {0, 1}, {1, -2}}  // trans B
  });
  auto& r2res_map = r2res_case[trans_b];
  // deduce states
  int32_t lrow = ds_a.get_dim(trans_a);
  int32_t lcol = ds_a.get_dim(1-trans_a);
  int32_t rrow = ds_b.get_dim(trans_b);
  int32_t rcol = ds_b.get_dim(1-trans_b);
  HT_ASSERT(lcol == rrow) << "Linear: tensor a.dimension[1] " << lcol 
    << " must be equal to tensor b.dimension[0] " << rrow;
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {-1, device_num/(lcol*lrow*rcol)}, {0, lrow}, {1, rcol}
  });
  // deduce order
  std::vector<int32_t> lorder = ds_a.get_order();
  std::vector<int32_t> rorder = ds_b.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_a.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  return {device_num, res_states, res_order};
}

void LinearOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const Tensor& a = inputs.at(0);
  const Tensor& b = inputs.at(1);
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  int32_t device_num = ds_a.get_device_num();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid()
            && ds_a.get_device_num() == ds_b.get_device_num())
            << "distributed states for Tensor a & Tensor b should be valid!";  
  Tensor bias;
  DistributedStates ds_bias;
  if (inputs.size() == 3) {
    bias = inputs.at(2);  
    ds_bias = bias->get_distributed_states();
    // check bias states
    if (trans_b()) { // bias shape = (b.shape[0], )
      HT_ASSERT(ds_b.get_dim(0) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 0 of b";
    } else { // bias shape = (b.shape[1], )
      HT_ASSERT(ds_b.get_dim(1) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 1 of b";
    }
  }
  // l,r to result states map  
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, 1}, {0, 0}, {1, -2}}, // no trans
    {{-1, 1}, {1, 0}, {0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a()];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, 0}, {0, -2}, {1, 1}}, // no trans
    {{-1, 0}, {0, 1}, {1, -2}}  // trans B
  });
  auto& r2res_map = r2res_case[trans_b()];
  // deduce states
  int32_t lrow = ds_a.get_dim(trans_a());
  int32_t lcol = ds_a.get_dim(1-trans_a());
  int32_t rrow = ds_b.get_dim(trans_b());
  int32_t rcol = ds_b.get_dim(1-trans_b());
  HT_ASSERT(lcol == rrow) << "Linear: tensor a.dimension[1] " << lcol 
    << " must be equal to tensor b.dimension[0] " << rrow;
  // if output states contains partial, then requires bias also should be partial
  HT_ASSERT(inputs.size() == 2 || lcol == ds_bias.get_dim(-2))
    << "Linear: partial in output states = " << lcol << " should be equal to partial of bias = " << ds_bias.get_dim(-2);
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {-1, device_num/(lcol*lrow*rcol)}, {0, lrow}, {1, rcol}
  });
  // deduce order
  std::vector<int32_t> lorder = ds_a.get_order();
  std::vector<int32_t> rorder = ds_b.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_a.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // set distributed states for result c
  Tensor& c = outputs.at(0);
  c->set_distributed_states({device_num, res_states, res_order});
}

void LinearOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  int32_t hetero_a = inputs_hetero_dim.at(0);
  int32_t hetero_b = inputs_hetero_dim.at(1);  
  if (trans_a() && (hetero_a == 0 || hetero_a == 1)) {
    hetero_a = 1 - hetero_a;
  }
  if (trans_b() && (hetero_b == 0 || hetero_b == 1)) {
    hetero_b = 1 - hetero_b;
  }
  int32_t hetero_res;
  if (hetero_a == NULL_HETERO_DIM) {
    HT_ASSERT(hetero_b == NULL_HETERO_DIM)
      << "Currently not support different union hetero type";
    hetero_res = NULL_HETERO_DIM;
  } else {
    if (hetero_a == -1 || hetero_b == -1) {
      if (hetero_a == -1) {
        HT_RUNTIME_ERROR << "not supported yet";
      }
      if (hetero_b == -1) {
        HT_ASSERT(hetero_a >= 0)
          << "hetero a and hetero b can't simutaneously be -1";
        hetero_res = hetero_a;
      }
    } else {
      HT_ASSERT(hetero_a == 1 - hetero_b)
        << "hetero a and hetero b should be opposite in this situation";
      hetero_res = -2;
    }
  }   
  outputs.at(0)->cur_ds_union().set_hetero_dim(hetero_res);
}

Tensor MakeLinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b), std::move(bias)};
  return Graph::MakeOp(
        std::make_shared<LinearOpImpl>(trans_a, trans_b),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeLinearOp(Tensor a, Tensor b, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<LinearOpImpl>(trans_a, trans_b),
        {std::move(a), std::move(b)},
        std::move(op_meta))->output(0);
}

void FusedColumnParallelLinearOpImpl::DoCompute(Operator& op,const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  auto& local_device = hetu::impl::comm::GetLocalDevice();

  const auto& src_group_union = op->input(0)->placement_group_union();
  const auto& dst_group_union = op->output(0)->placement_group_union();
  auto src_ds_union = op->input(0)->cur_ds_union();
  size_t src_union_idx = 0, dst_union_idx = 0;
  if (src_group_union.has(local_device)) {
    src_union_idx = src_group_union.get_index(local_device);
  }
  if (dst_group_union.has(local_device)) {
    dst_union_idx = dst_group_union.get_index(local_device);
  }
  int32_t local_device_idx = dst_group_union.get(dst_union_idx).get_index(local_device);
  DeviceGroup comm_group = _allgather_dst_ds.get_devices_by_dim(-1, local_device_idx, dst_group_union.get(dst_union_idx));
  int32_t gather_dim = src_ds_union.get(src_union_idx).get_split_dim(_allgather_dst_ds);
  
  // 策略1: p2p ring-exchange实现
  // int tp_size = comm_group.num_devices();
  // int local_world_rank = hetu::impl::comm::DeviceToWorldRank(local_device);
  // // 目前认为的local, prev, next的tp_rank计算方法，即认为一个tp组的device rank号是连续的
  // int local_tp_rank = local_world_rank % tp_size;
  // int prev_tp_rank = (local_world_rank - 1 + tp_size) % tp_size;
  // int next_tp_rank = (local_world_rank + 1 + tp_size) % tp_size;
  // int prev_world_rank = prev_tp_rank + (local_world_rank / tp_size) * tp_size;
  // int next_world_rank = next_tp_rank + (local_world_rank / tp_size) * tp_size;

  // auto used_ranks = dynamic_cast<ExecutableGraph&>(op->graph()).GetUsedRanks();
  // auto& nccl_comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(used_ranks, Stream(local_device, kP2PStream));

  // std::vector<std::unique_ptr<hetu::impl::CUDAEvent>> comp_events;
  // std::vector<std::unique_ptr<hetu::impl::CUDAEvent>> comm_events;
  // for(int i = 0; i < tp_size-1; ++i) {
  //   comp_events.push_back(std::make_unique<hetu::impl::CUDAEvent>(local_device));
  //   comm_events.push_back(std::make_unique<hetu::impl::CUDAEvent>(local_device));
  // }

  // auto src_shape = inputs[0]->shape();
  // NDArray p2p_buffer1 = NDArray::empty(src_shape, op->instantiation_ctx().placement,
  //   op->input(0)->dtype(),  kP2PStream);  // 偶数轮次（0，2，4，...）作为发送buffer，奇数轮次作为接收buffer
  // NDArray p2p_buffer2 = NDArray::empty(src_shape, op->instantiation_ctx().placement,
  //   op->input(0)->dtype(),  kP2PStream);  // 偶数轮次作为接收buffer，奇数轮次作为发送buffer

  // // 确保之前的norm操作进行完后再做p2p通信
  // auto norm_event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
  // norm_event->Record(Stream(local_device, kComputingStream));
  // norm_event->Block(Stream(local_device, kP2PStream));

  // // 每轮向prev发送数据，接收来自next的数据
  // for(int i = 0; i < tp_size; ++i) {
  //   if(i < tp_size - 1) {
  //     // 需要发送和接收数据
  //     // 等上一轮计算结束
  //     if(i > 0) {
  //       comp_events[i-1]->Block(Stream(local_device, kP2PStream));
  //     }
  //     std::vector<hetu::impl::comm::CommTask> ring_send_recv_tasks;
  //     ring_send_recv_tasks.push_back(nccl_comm_group->ISend((i % 2 == 0)? ((i == 0)? inputs[0]: p2p_buffer1): p2p_buffer2, prev_world_rank));
  //     ring_send_recv_tasks.push_back(nccl_comm_group->IRecv((i % 2 == 0)? p2p_buffer2: p2p_buffer1, next_world_rank));
  //     nccl_comm_group->BatchedISendIRecv(ring_send_recv_tasks);
  //     comm_events[i]->Record(Stream(local_device, kP2PStream));
  //   }
    
  //   // GEMM计算
  //   // 等上一轮通信结束
  //   if(i > 0) {
  //     comm_events[i-1]->Block(Stream(local_device, kComputingStream));
  //   }
  //   int comp_chunk_id = (local_tp_rank + i) % tp_size;
  //   NDArray output_slice = NDArray::slice(outputs.at(0), 
  //                   {src_shape[0] * comp_chunk_id, 0}, {src_shape[0], outputs.at(0)->shape(1)}, kComputingStream);
  //   if(inputs.size() == 2) {
  //     NDArray::linear((i % 2 == 0)? ((i == 0)? inputs[0]: p2p_buffer1): p2p_buffer2, inputs.at(1), 
  //                     NDArray(), trans_a(), trans_b(),
  //                     kComputingStream, output_slice);
  //   } else if(inputs.size() == 3) {
  //     NDArray::linear((i % 2 == 0)? ((i == 0)? inputs[0]: p2p_buffer1): p2p_buffer2, inputs.at(1), 
  //                     inputs.at(2), trans_a(), trans_b(),
  //                     kComputingStream, output_slice);
  //   }
  //   if(i < tp_size - 2) comp_events[i]->Record(Stream(local_device, kComputingStream));
  // }

  
  // 策略2: allgather + gemm实现
  auto allgather_comm_event = hetu::impl::CUDAEvent(local_device);

  auto src_shape = inputs[0]->shape();
  auto all_gather_dst_shape = src_shape;
  all_gather_dst_shape[0] = src_shape[0] * _allgather_dst_ds.get_dim(-1);
  NDArray allgather_output = NDArray::empty(
    all_gather_dst_shape, 
    op->instantiation_ctx().placement,
    op->input(0)->dtype(),
    kCollectiveStream
  );

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllGather, inputs.at(0), allgather_output, 
                                  comm_group, gather_dim, Stream(local_device, kCollectiveStream));
  allgather_comm_event.Record(Stream(local_device, kCollectiveStream));

  allgather_comm_event.Block(Stream(local_device, kComputingStream));
  if (inputs.size() == 2)
    NDArray::linear(allgather_output, inputs.at(1), NDArray(), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
  else if (inputs.size() == 3)
    NDArray::linear(allgather_output, inputs.at(1), inputs.at(2), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList FusedColumnParallelLinearOpImpl::DoGradient(Operator& op,const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);

  // 暂时不考虑不同的op->requires_grad()
  if(!op->requires_grad(0)) {
    if(op->num_inputs() == 2) {
      return {Tensor(), Tensor()};
    }
    else if(op->num_inputs() == 3) {
      return {Tensor(), Tensor(), Tensor()};
    }
  }

  if(op->num_inputs() == 2) {
    TensorList grad_inputs = 
        MakeFusedColumnParallelLinearGradientOp(grad_c, a, b, trans_a(), trans_b(), 
          _allgather_dst_ds, op->grad_op_meta().set_name(op->grad_name()));
    Tensor grad_a = grad_inputs.at(0);
    Tensor grad_b = grad_inputs.at(1);
    return {grad_a, grad_b};
  }
  else if(op->num_inputs() == 3) {
    TensorList grad_inputs = 
        MakeFusedColumnParallelLinearGradientOp(grad_c, a, b, op->input(2), trans_a(), trans_b(), 
          _allgather_dst_ds, op->grad_op_meta().set_name(op->grad_name()));
    Tensor grad_a = grad_inputs.at(0);
    Tensor grad_b = grad_inputs.at(1);
    Tensor grad_bias = grad_inputs.at(2);
    return {grad_a, grad_b, grad_bias};
  }
}

HTShapeList FusedColumnParallelLinearOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  HT_ASSERT(false) << "Currently don't use this";
  return {};
}

void FusedColumnParallelLinearOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const Tensor& a = inputs.at(0);
  const Tensor& b = inputs.at(1);
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  int32_t device_num = ds_a.get_device_num();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid()
            && ds_a.get_device_num() == ds_b.get_device_num())
            << "distributed states for Tensor a & Tensor b should be valid!";  
  Tensor bias;
  DistributedStates ds_bias;
  if (inputs.size() == 3) {
    bias = inputs.at(2);  
    ds_bias = bias->get_distributed_states();
    // check bias states
    if (trans_b()) { // bias shape = (b.shape[0], )
      HT_ASSERT(ds_b.get_dim(0) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 0 of b";
    } else { // bias shape = (b.shape[1], )
      HT_ASSERT(ds_b.get_dim(1) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 1 of b";
    }
  }
  // l,r to result states map  
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, 1}, {0, 0}, {1, -2}}, // no trans
    {{-1, 1}, {1, 0}, {0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a()];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, 0}, {0, -2}, {1, 1}}, // no trans
    {{-1, 0}, {0, 1}, {1, -2}}  // trans B
  });
  auto& r2res_map = r2res_case[trans_b()];
  // deduce states
  int32_t lrow = ds_b.get_dim(-1);  // 此处用ds_b[-1]替代ds_a[trans_a()]
  int32_t lcol = ds_a.get_dim(1-trans_a());
  int32_t rrow = ds_b.get_dim(trans_b());
  int32_t rcol = ds_b.get_dim(1-trans_b());
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {-1, device_num/(lcol*lrow*rcol)}, {0, lrow}, {1, rcol}
  });
  // deduce order
  std::vector<int32_t> lorder = {};  // 此处应计算输入数据按照tp(sp)维度allgather后的order
  if(lrow > 1) {
    // 有cp
    lorder.push_back(0);
  }
  if(rcol > 1) {
    // 有tp
    lorder.push_back(-1);
  }
  std::vector<int32_t> rorder = ds_b.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && rcol > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // set distributed states for result c
  Tensor& c = outputs.at(0);
  c->set_distributed_states({device_num, res_states, res_order});

  _allgather_dst_ds = DistributedStates(
    device_num, 
    {{-2, 1}, {-1, ds_b.get_dim(1-trans_b())}, {0, ds_b.get_dim(-1)}, {1, 1}},
    lorder
  );
}

void FusedColumnParallelLinearOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  HT_ASSERT(false) << "Currently don't support hetero";
}

Tensor MakeFusedColumnParallelLinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b), std::move(bias)};
  return Graph::MakeOp(
        std::make_shared<FusedColumnParallelLinearOpImpl>(trans_a, trans_b),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeFusedColumnParallelLinearOp(Tensor a, Tensor b, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<FusedColumnParallelLinearOpImpl>(trans_a, trans_b),
        {std::move(a), std::move(b)},
        std::move(op_meta))->output(0);
}


void FusedColumnParallelLinearGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {  
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  // 创建事件用于同步
  auto dgrad_comp_event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
  auto allgather_comm_event = std::make_unique<hetu::impl::CUDAEvent>(local_device);

  // 分配allgather buffer
  // 目前src_group和dst_group都是一样的，所以此处没必要纠结具体的src_group和dst_group的求法
  const auto& src_group_union = op->input(1)->placement_group_union();
  const auto& dst_group_union = op->input(0)->placement_group_union();
  auto src_ds_union = op->input(1)->cur_ds_union();
  size_t src_union_idx = 0, dst_union_idx = 0;
  if (src_group_union.has(local_device)) {
    src_union_idx = src_group_union.get_index(local_device);
  }
  if (dst_group_union.has(local_device)) {
    dst_union_idx = dst_group_union.get_index(local_device);
  }
  int32_t local_device_idx = dst_group_union.get(dst_union_idx).get_index(local_device);
  DeviceGroup comm_group = _allgather_dst_ds.get_devices_by_dim(-1, local_device_idx, dst_group_union.get(dst_union_idx));
  int32_t gather_dim = src_ds_union.get(src_union_idx).get_split_dim(_allgather_dst_ds);
  
  auto src_shape = inputs[1]->shape();
  auto all_gather_dst_shape = src_shape;
  all_gather_dst_shape[0] = src_shape[0] * _allgather_dst_ds.get_dim(-1);
  NDArray allgather_output = NDArray::empty(
    all_gather_dst_shape, 
    op->instantiation_ctx().placement,
    op->input(1)->dtype(),
    kCollectiveStream
  );

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllGather, inputs.at(1), allgather_output, 
                                  comm_group, gather_dim, Stream(local_device, kCollectiveStream));
  allgather_comm_event->Record(Stream(local_device, kCollectiveStream));
  // dgrad
  const auto& grad_c = inputs.at(0);
  const auto& a = inputs.at(1);
  const auto& b = inputs.at(2);
  HT_ASSERT(!trans_a() && trans_b());
  NDArray tmp;
  NDArray dgrad_output = NDArray::linear(grad_c, b, NDArray(), false, false,
                    op->instantiation_ctx().stream_index, tmp);
  dgrad_comp_event->Record(Stream(local_device, kComputingStream));
  // dgrad rs
  // 认为rs的comm_group与ag的comm_group一样
  dgrad_comp_event->Block(Stream(local_device, kCollectiveStream));
  int32_t scatter_dim = op->output(0)->get_distributed_states().get_split_dim(_allgather_dst_ds);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReduceScatter, dgrad_output, outputs.at(0), 
                                  ReductionType::SUM, comm_group, scatter_dim, false, Stream(local_device, kCollectiveStream));

  // wgrad
  // 在此之前应该要等待allgather完成
  allgather_comm_event->Block(Stream(local_device, kComputingStream));
  NDArray::linear(grad_c, allgather_output, NDArray(), true, false,
                    op->instantiation_ctx().stream_index, outputs.at(1));

  if(op->num_inputs() == 4) {
    // bias grad
    NDArray::reduce(grad_c, ReductionType::SUM, {0}, false,
                  op->instantiation_ctx().stream_index, outputs.at(2));
  }

  // 下一个计算任务开始前要等待dgrad_rs结束
  auto dgrad_rs_end_event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
  dgrad_rs_end_event->Record(Stream(local_device, kCollectiveStream));
  dgrad_rs_end_event->Block(Stream(local_device, kComputingStream));
}

HTShapeList FusedColumnParallelLinearGradientOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  HT_ASSERT(false) << "Currently don't use this";
  return {};
}

void FusedColumnParallelLinearGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const Tensor& grad_c = inputs.at(0);
  const Tensor& a = inputs.at(1);
  const Tensor& b = inputs.at(2);
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  outputs.at(0)->set_distributed_states(ds_a);
  // 注意这里weight_grad的ds不一定与weight的ds相同
  outputs.at(1)->set_distributed_states(deduce_states(grad_c->get_distributed_states(), _allgather_dst_ds, true, false));

  if(inputs.size() == 4) {
    outputs.at(2)->set_distributed_states(ReduceOpImpl::StatesForDistributedReduce(grad_c, {0}, {false}));
  }
}

void FusedColumnParallelLinearGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  HT_ASSERT(false) << "Currently don't support hetero";
}

TensorList MakeFusedColumnParallelLinearGradientOp(Tensor grad_c, Tensor a, Tensor b, bool trans_a,
                    bool trans_b, DistributedStates allgather_dst_ds , OpMeta op_meta) {
  TensorList inputs = {std::move(grad_c), std::move(a), std::move(b)};
  return Graph::MakeOp(
        std::make_shared<FusedColumnParallelLinearGradientOpImpl>(trans_a, trans_b, allgather_dst_ds),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

TensorList MakeFusedColumnParallelLinearGradientOp(Tensor grad_c, Tensor a, Tensor b, Tensor bias, bool trans_a,
                    bool trans_b, DistributedStates allgather_dst_ds, OpMeta op_meta) {
  TensorList inputs = {std::move(grad_c), std::move(a), std::move(b), std::move(bias)};
  return Graph::MakeOp(
        std::make_shared<FusedColumnParallelLinearGradientOpImpl>(trans_a, trans_b, allgather_dst_ds),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
