#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/tensor.h"
#include "hetu/autograd/distributed_states.h"

namespace hetu {
namespace autograd {

class AllReduceOpDef;
class AllReduceOp;
class P2PSendOpDef;
class P2PSendOp;
class P2PRecvOpDef;
class P2PRecvOp;
class CommOpDef;
class CommOp;

// tensor1 = op1(xxx)
// tensor2 = comm_op(tensor1, distributed_state) // distributed_state是给定的op2 input所需的切分状态
// tensor3 = op2(tensor2) 

class CommOpDef : public OperatorDef {
 private:
  friend class CommOp;
  struct constructor_access_key {};

 public:
  CommOpDef(const constructor_access_key&, Tensor input, 
            DistributedStates dst_distributed_states, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(CommOp), {input}, op_meta), _dst_distributed_states(dst_distributed_states) {
    // op1 (tensor1) -> comm_op (tensor2) -> op2 (tensor3), 用具体的通信算子替换comm_op 
    // 1. 根据input->states+dst_states来判断要插入什么样的通信算子
    auto src_distributed_states = input->get_distributed_states();
    if (src_distributed_states.check_allreduce(dst_distributed_states)) {
      _comm_type = ALL_REDUCE_OP;
      HT_LOG_DEBUG << "ALL_REDUCE_OP";
    } else if (src_distributed_states.check_allgather(dst_distributed_states)) {
      _comm_type = ALL_GATHER_OP;
      HT_LOG_DEBUG << "ALL_GATHER_OP";
    } else if (src_distributed_states.check_reducescatter(dst_distributed_states)) {
      _comm_type = REDUCE_SCATTER_OP;
      HT_LOG_DEBUG << "REDUCE_SCATTER_OP";
    } else if (src_distributed_states.check_boradcast(dst_distributed_states)) {
      _comm_type = BROADCAST_OP;
      HT_LOG_DEBUG << "BROADCAST_OP";
    } else if (src_distributed_states.check_reduce(dst_distributed_states)) {
      _comm_type = REDUCE_OP;
      HT_LOG_DEBUG << "REDUCE_OP";
    } else {
      // other case: 非集合通信, 部分device之间p2p通信
      _comm_type = P2P_OP;
      HT_LOG_DEBUG << "P2P_OP";
    }

    // 2. comm_op -> op2, 其中op2所需的tensor分片已经由集合通信转化完毕, 接下来在op2的do compute里确定output的distributed_states
    // add output
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return COMM_OP;
  }

  inline DistributedStates get_dst_distributed_states() const {
    return _dst_distributed_states;
  }

  inline uint64_t get_comm_type() const {
    return _comm_type;
  }
  
 protected:
  uint64_t _comm_type;
  DistributedStates _dst_distributed_states;
};

class CommOp final : public OpWrapper<CommOpDef> {
 public:
  CommOp(Tensor input, DistributedStates& dst_distributed_states, 
         const OpMeta& op_meta = OpMeta())
  : OpWrapper<CommOpDef>(make_ptr<CommOpDef>(
    CommOpDef::constructor_access_key(), input, dst_distributed_states, op_meta)) {}
};

class AllReduceOpDef : public OperatorDef {
 private:
  friend class AllReduceOp;
  struct constrcutor_access_key {};

 public:
  AllReduceOpDef(const constrcutor_access_key&, Tensor input,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AllReduceOpDef), {input}, op_meta) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << "AllReduce requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return ALL_REDUCE_OP;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class AllReduceOp final : public OpWrapper<AllReduceOpDef> {
 public:
  AllReduceOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AllReduceOpDef>(make_ptr<AllReduceOpDef>(
      AllReduceOpDef::constrcutor_access_key(), input, op_meta)) {}
};

class P2PSendOpDef : public OperatorDef {
 private:
  friend class P2PSendOp;
  struct constrcutor_access_key {};

 public:
  P2PSendOpDef(const constrcutor_access_key&, Tensor input,
               const DeviceGroup& dst_group, int dst_device_index = -1, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PSendOp), {input}, op_meta), _dst_group(dst_group), _dst_device_index(dst_device_index) {
    HT_ASSERT(!dst_group.empty())
      << "Please provide the \"dst_group\" argument to indicate "
      << "the destination devices for P2PSend";
    HT_ASSERT(device_group().empty() ||
              device_group().num_devices() == dst_group.num_devices())
      << "Currently we require equal data parallelism degree across "
      << "P2P communication. Got " << device_group() << " vs. " << dst_group;
  }

  // Walkaround: bind the send and recv op
  void BindRecvOp(const P2PRecvOp& recv_op);

  const P2PRecvOp& recv_op() const;

  P2PRecvOp& recv_op();

  int index_in_group() {
    return _index_in_group;
  }

  void set_index_in_group(int index) {
    // 要么是在map to local device的时候已经赋值好了, 要么就是在这里赋值(用于distributed tensor中非local device的p2p op的赋值)
    HT_ASSERT(_index_in_group == -1) << "only allow set when _index_in_group = -1!";
    _index_in_group = index;
  }  

  const DeviceGroup& dst_group() const {
    return _dst_group;
  }

  int dst_device_index() {
    return _dst_device_index;
  }

  bool is_distributed_tensor_send_op() {
    return _dst_device_index != -1;
  }  

  OpList& send_recv_topo() {
    return _send_recv_topo;
  }

  void set_send_recv_topo(OpList& send_recv_topo) {
    _send_recv_topo = send_recv_topo;
  }  

  uint64_t op_indicator() const noexcept {
    return PEER_TO_PEER_SEND_OP;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  OpList _send_recv_topo{};
  DeviceGroup _dst_group;
  int _dst_device_index{-1}; // for distributed tensor p2p
  int _index_in_group{-1}; // for pipeline p2p
  // The `_recv_op` member is wrapped into a unique pointer since
  // the forward declared `P2PRecvOp` class has incomplete type now.
  std::unique_ptr<P2PRecvOp> _recv_op;
};

class P2PSendOp final : public OpWrapper<P2PSendOpDef> {
 public:
  P2PSendOp(Tensor input, const DeviceGroup& dst_group, 
            int dst_device_index = -1, const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PSendOpDef>(make_ptr<P2PSendOpDef>(
      P2PSendOpDef::constrcutor_access_key(), input, dst_group, dst_device_index, op_meta)) {}
};

class P2PRecvOpDef : public OperatorDef {
 private:
  friend class P2PRecvOp;
  struct constrcutor_access_key {};

 public:
  P2PRecvOpDef(const constrcutor_access_key&, const DeviceGroup& src_group, 
               DataType dtype, const HTShape& shape = HTShape(), int src_device_index = -1,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PRecvOp), TensorList(), op_meta),
    _src_group(src_group), _src_device_index(src_device_index) {
    HT_ASSERT(!src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(device_group().empty() ||
              src_group.num_devices() == device_group().num_devices())
      << "Currently we require equal data parallelism degree across "
      << "P2P communication. Got " << src_group << " vs. " << device_group();
    HT_ASSERT(dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
    AddOutput(NDArrayMeta().set_dtype(dtype).set_shape(shape));
  }

  // Walkaround: bind the send and recv op
  void BindSendOp(const P2PSendOp& send_op);

  const P2PSendOp& send_op() const;

  P2PSendOp& send_op();

  int index_in_group() {
    return _index_in_group;
  }

  void set_index_in_group(int index) {
    // 要么是在map to local device的时候已经赋值好了, 要么就是在这里赋值(用于distributed tensor中非local device的p2p op的赋值)
    HT_ASSERT(_index_in_group == -1) << "only allow set when _index_in_group = -1!";
    _index_in_group = index;
  }

  const DeviceGroup& src_group() const {
    return _src_group;
  }

  int src_device_index() {
    return _src_device_index;
  }

  bool is_distributed_tensor_recv_op() {
    return _src_device_index != -1;
  }

  OpList& send_recv_topo() {
    return _send_recv_topo;
  }

  void set_send_recv_topo(OpList& send_recv_topo) {
    _send_recv_topo = send_recv_topo;
  }

  uint64_t op_indicator() const noexcept {
    return PEER_TO_PEER_RECV_OP;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  OpList _linked_ops{};
  OpList _send_recv_topo{};
  DeviceGroup _src_group;
  int _src_device_index{-1}; // for distributed tensor p2p
  int _index_in_group{-1}; // for pipeline p2p
  std::unique_ptr<P2PSendOp> _send_op;
};

class P2PRecvOp final : public OpWrapper<P2PRecvOpDef> {
 public:
  P2PRecvOp(const DeviceGroup& src_group, DataType dtype,
            const HTShape& shape = HTShape(), int src_device_index = -1, 
            const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PRecvOpDef>(
      make_ptr<P2PRecvOpDef>(P2PRecvOpDef::constrcutor_access_key(), src_group,
                             dtype, shape, src_device_index, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
