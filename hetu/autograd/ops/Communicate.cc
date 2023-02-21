#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

bool CommOpDef::DoMapToParallelDevices(const DeviceGroup& placement_group) {
  auto local_device = GetLocalDevice();
  // 如果input是由comm_op产生的, 则合并这两个comm_op(否则相邻两个comm_op在SplitOp推导shape中会有问题, 暂时还没找到具体原因)
  auto& input_op = _inputs[0]->producer();
  if (is_comm_op(input_op)) {
    HT_LOG_DEBUG << local_device << ": " << name() << ": replace input from " << _inputs[0] << " to " << input_op->input(0);     
    ReplaceInput(0, input_op->input(0));
    HT_LOG_DEBUG << local_device << ": " << name() << ": inputs: " << _inputs << ", outputs: " << _outputs;
    // for (int i = 0; i <)
    get_comm_type();
  }
  return OperatorDef::DoMapToParallelDevices(placement_group);  
}

// TODO: infer shape的代码逻辑需要纠正
HTShapeList CommOpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape& input_shape = input_shapes.at(0);

  Tensor& input = _inputs[0];
  auto src_ds = input->get_distributed_states();
  auto dst_ds = get_dst_distributed_states();

  HTShape shape; shape.reserve(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  
  return {shape};
}

TensorList CommOpDef::DoGradient(const TensorList& grad_outputs) {
  Tensor& input = _inputs[0];
  auto ds_input = input->get_distributed_states();
  Tensor& output = _outputs[0];
  auto ds_output = output->get_distributed_states();
  const Tensor& grad_output = grad_outputs.at(0);
  auto ds_grad_output = grad_output->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_output.is_valid())
           << "distributed states for input and output tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_output.get_device_num())
           << "distributed states for input and output tensor must be matched!";
  // HT_ASSERT() // TODO: check ds_grad_output and ds_output must be same
  DistributedStates ds_grad_input(ds_input);
  if (ds_grad_input.get_dim(-2) > 1) { // partial->duplicate
    std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
    auto res_states = ds_grad_input.combine_states(src2dst);
    auto res_order = ds_grad_input.combine_order(src2dst);
    auto device_num = ds_grad_input.get_device_num();
    ds_grad_input.set_distributed_states({device_num, res_states, res_order});
  }
  Tensor grad_input = CommOp(grad_output, ds_grad_input, OpMeta().set_name("grad_" + name()))->output(0);
  return {grad_input};
}

void CommOpDef::ForwardDeduceStates() {
  Tensor& input = _inputs[0];
  DistributedStates ds_input = input->get_distributed_states();
  DistributedStates ds_dst = get_dst_distributed_states();
  // TODO: check states/order between src and dst
  HT_ASSERT(ds_input.is_valid() && ds_dst.is_valid())
           << "distributed states for input and dst tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_dst.get_device_num())
           << "cannot convert src distributed states to unpaired dst distributed states!";
  Tensor& output = _outputs[0];
  output->set_distributed_states(ds_dst);
}

DistributedStates CommOpDef::BackwardDeduceStates(int32_t index) {
  HT_ASSERT(index == 0) << "grad index in CommOp must be equal to 0!";
  Tensor& input = _inputs[0];
  auto ds_input = input->get_distributed_states();
  Tensor& output = _outputs[0];
  auto ds_output = output->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_output.is_valid())
           << "distributed states for input and output tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_output.get_device_num())
           << "distributed states for input and output tensor must be matched!";
  DistributedStates ds_input_grad(ds_input);
  if (ds_input_grad.get_dim(-2) > 1) { // partial->duplicate
    std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
    auto new_states = ds_input_grad.combine_states(src2dst);
    auto new_order = ds_input_grad.combine_order(src2dst);
    auto device_num = ds_input_grad.get_device_num();
    ds_input_grad.set_distributed_states({device_num, new_states, new_order});
  }
  return ds_input_grad;
}

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
  // HT_LOG_INFO << "all_reduce_op inputs: " << inputs << ", outputs: " << outputs;                                
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
