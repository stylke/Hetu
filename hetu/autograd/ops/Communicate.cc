#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void P2PSendOpDef::BindRecvOp(const P2PRecvOp& recv_op) {
  HT_ASSERT(_recv_op == nullptr) << "Try to bind P2PRecvOp twice";
  _recv_op = std::make_unique<P2PRecvOp>(recv_op);
}

const P2PRecvOp& P2PSendOpDef::recv_op() const {
  HT_ASSERT(_recv_op != nullptr) << "P2PRecvOp is not bound yet";
  return *_recv_op;
}

P2PRecvOp& P2PSendOpDef::recv_op() {
  HT_ASSERT(_recv_op != nullptr) << "P2PRecvOp is not bound yet";
  return *_recv_op;
}

void P2PRecvOpDef::BindSendOp(const P2PSendOp& send_op) {
  HT_ASSERT(_send_op == nullptr) << "Try to bind P2PSendOp twice";
  _send_op = std::make_unique<P2PSendOp>(send_op);
}

const P2PSendOp& P2PRecvOpDef::send_op() const {
  HT_ASSERT(_send_op != nullptr) << "P2PSendOp is not bound yet";
  ;
  return *_send_op;
}

P2PSendOp& P2PRecvOpDef::send_op() {
  HT_ASSERT(_send_op != nullptr) << "P2PSendOp is not bound yet";
  ;
  return *_send_op;
}

bool AllReduceOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call AllReduce with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList AllReduceOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support in-place?
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::AllReduce, inputs.at(0),
                                  outputs.at(0), placement_group(), stream());
  return outputs;
}

HTShapeList AllReduceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

bool P2PSendOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  HT_ASSERT(pg.num_devices() == _dst_group.num_devices())
    << "Currently we require equal data parallelism degree across "
    << "P2P communication. Got " << pg << " vs. " << _dst_group;

  _placement_group = pg;
  return true;
}

bool P2PSendOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  _index_in_group = _placement_group.get_index(placement);
  HT_ASSERT(is_distributed_tensor_send_op() || !is_distributed_tensor_send_op() && _dst_group.get(_index_in_group) != placement)
    << "Pipeline p2p send op: source and destination are the same (" << placement << ")";
  if (!is_distributed_tensor_send_op()) {
    _dst_device_index = _index_in_group;
  }  
  return OperatorDef::DoPlaceToLocalDevice(placement, stream_id);
}

NDArrayList P2PSendOpDef::DoCompute(const NDArrayList& inputs,
                                    RuntimeContext& ctx) {
  NDArray input = inputs.at(0);
  HT_ASSERT(input->dtype() == _inputs[0]->dtype())
    << "Data type mismatched for P2P communication: " << input->dtype()
    << " vs. " << _inputs[0]->dtype();

  // TODO: sending the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray send_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  auto* ptr = send_shape->data_ptr<int64_t>();
  ptr[0] = static_cast<int64_t>(input->ndim());
  std::copy(input->shape().begin(), input->shape().end(), ptr + 1);
  hetu::impl::P2PSendCpu(send_shape, _dst_group.get(_dst_device_index),
                         Stream(Device(kCPU), kBlockingStream));

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::P2PSend, input,
                                  _dst_group.get(_dst_device_index), stream());
  return NDArrayList();
}

bool P2PRecvOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  HT_ASSERT(pg.num_devices() == _src_group.num_devices())
    << "Currently we require equal data parallelism degree across "
    << "P2P communication. Got " << _src_group << " vs. " << pg;

  _placement_group = pg;
  // TODO: set the parallel statuses of output
  for (auto& output : _outputs) {
    output->set_placement_group(pg);
  }  
  return true;
}

bool P2PRecvOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  _index_in_group = _placement_group.get_index(placement);
  HT_ASSERT(is_distributed_tensor_recv_op() || !is_distributed_tensor_recv_op() && _src_group.get(_index_in_group) != placement)
    << "Pipeline p2p recv op: source and destination are the same (" << placement << ")";
  if (!is_distributed_tensor_recv_op()) {
    _src_device_index = _index_in_group;
  }
  return OperatorDef::DoPlaceToLocalDevice(placement, stream_id);
}

NDArrayList P2PRecvOpDef::DoCompute(const NDArrayList& inputs,
                                    RuntimeContext& ctx) {
  // TODO: receiving the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray recv_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  hetu::impl::P2PRecvCpu(recv_shape, _src_group.get(_src_device_index),
                         Stream(Device(kCPU), kBlockingStream));
  auto* ptr = recv_shape->data_ptr<int64_t>();
  HTShape shape(ptr + 1, ptr + 1 + ptr[0]);
  NDArray output = NDArray::empty(shape, placement(), _outputs[0]->dtype());

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::P2PRecv, output,
                                  _src_group.get(_src_device_index), stream());
  return {output};
}

} // namespace autograd
} // namespace hetu
