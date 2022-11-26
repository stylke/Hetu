#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class AllReduceOpDef;
class AllReduceOp;
class P2PSendOpDef;
class P2PSendOp;
class P2PRecvOpDef;
class P2PRecvOp;

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
               const DeviceGroup& dst_group, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PSendOp), {input}, op_meta), _dst_group(dst_group) {
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

  const DeviceGroup& dst_group() const {
    return _dst_group;
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

  DeviceGroup _dst_group;
  int _index_in_group{-1};
  // The `_recv_op` member is wrapped into a unique pointer since
  // the forward declared `P2PRecvOp` class has incomplete type now.
  std::unique_ptr<P2PRecvOp> _recv_op;
};

class P2PSendOp final : public OpWrapper<P2PSendOpDef> {
 public:
  P2PSendOp(Tensor input, const DeviceGroup& dst_group,
            const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PSendOpDef>(make_ptr<P2PSendOpDef>(
      P2PSendOpDef::constrcutor_access_key(), input, dst_group, op_meta)) {}
};

class P2PRecvOpDef : public OperatorDef {
 private:
  friend class P2PRecvOp;
  struct constrcutor_access_key {};

 public:
  P2PRecvOpDef(const constrcutor_access_key&, const DeviceGroup& src_group,
               DataType dtype, const HTShape& shape = HTShape(),
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PRecvOp), TensorList(), op_meta),
    _src_group(src_group) {
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

  const DeviceGroup& src_group() const {
    return _src_group;
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

  DeviceGroup _src_group;
  int _index_in_group{-1};
  std::unique_ptr<P2PSendOp> _send_op;
};

class P2PRecvOp final : public OpWrapper<P2PRecvOpDef> {
 public:
  P2PRecvOp(const DeviceGroup& src_group, DataType dtype,
            const HTShape& shape = HTShape(), const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PRecvOpDef>(
      make_ptr<P2PRecvOpDef>(P2PRecvOpDef::constrcutor_access_key(), src_group,
                             dtype, shape, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
