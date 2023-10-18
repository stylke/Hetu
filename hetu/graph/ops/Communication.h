#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

class CommOpImpl;
class AllReduceOpImpl;
class P2PSendOpImpl;
class P2PRecvOpImpl;
class BatchedISendIRecvOpImpl;
class AllGatherOpImpl;
class ReduceScatterOpImpl;
class ScatterOpImpl;

class CommOpImpl final: public OpInterface {
 public:
  CommOpImpl(DistributedStates dst_ds, DeviceGroup dst_group = DeviceGroup())
  : OpInterface(quote(CommOp)), _dst_ds(dst_ds), _dst_group(dst_group) {}

  uint64_t op_indicator() const noexcept override {
    return COMM_OP;
  }  

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;                              

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const {}

 public: 
  const DistributedStates& get_dst_distributed_states() const {
    return _dst_ds;
  }

  const DeviceGroup& src_group(Operator& op) const {
    return op->input(0)->placement_group();
  }

  const DeviceGroup& dst_group(Operator& op) const {
    if (_dst_group.empty()) {
      return op->input(0)->placement_group();
    } else {
      return _dst_group;
    }
  }

  bool is_intra_group(Operator& op) const {
    return !is_inter_group(op);
  }

  bool is_inter_group(Operator& op) const {
    return src_group(op) != dst_group(op);
  }

  uint64_t get_comm_type(Operator& op);

  DeviceGroup get_devices_by_dim(Operator& op, int32_t dim) const; 

 protected:
  uint64_t _comm_type{UNKNOWN_OP};
  DistributedStates _dst_ds;
  DeviceGroup _dst_group;
};

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, 
                  DeviceGroup dst_group, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, 
                  OpMeta op_meta = OpMeta());

class AllReduceOpImpl final : public OpInterface {
 public:
  AllReduceOpImpl(const DeviceGroup& comm_group,
                  const DeviceGroup& device_group = DeviceGroup())
  : OpInterface(quote(AllReduceOp)), _comm_group(comm_group) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
             << "AllReduce requires two or more comm devices. Got " << _comm_group;
    if (!device_group.empty()) {
      for (int i = 0; i < comm_group.num_devices(); i++) {
        HT_ASSERT(device_group.contains(comm_group.get(i))) 
          << "AllReduceOp: device in comm_group: " << comm_group.get(i) 
          << " must in device_group: " << device_group;
      }
    }
  }

  uint64_t op_indicator() const noexcept override {
    return ALL_REDUCE_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  const DeviceGroup& comm_group() const {
    return _comm_group;
  }

 protected:
  DeviceGroup _comm_group;
};

Tensor MakeAllReduceOp(Tensor input, const DeviceGroup& comm_group, 
                       OpMeta op_meta = OpMeta());

class P2PSendOpImpl final : public OpInterface {
 public:
  P2PSendOpImpl(const DeviceGroup& dst_group, int dst_device_index = -1, 
                const DeviceGroup& device_group = DeviceGroup())
  : OpInterface(quote(P2PSendOp)), _dst_group(dst_group), 
    _dst_device_index(dst_device_index) {
    HT_ASSERT(!dst_group.empty())
      << "Please provide the \"dst_group\" argument to indicate "
      << "the destination devices for P2PSend";
    HT_ASSERT(device_group.empty() ||
              device_group.num_devices() == dst_group.num_devices())
      << "Currently we require equal tensor parallelism degree across "
      << "P2P communication. Got " << device_group << " vs. " << dst_group;
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_SEND_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
                        
 public:
  const DeviceGroup& dst_group() const {
    return _dst_group;
  }

  int dst_device_index() const {
    return _dst_device_index;
  }  

 protected:
  DeviceGroup _dst_group;
  int _dst_device_index{-1};
};

Tensor MakeP2PSendOp(Tensor input, const DeviceGroup& dst_group, 
                     int dst_device_index = -1, OpMeta op_meta = OpMeta());

class P2PRecvOpImpl final : public OpInterface {
 public:
  P2PRecvOpImpl(const DeviceGroup& src_group, DataType dtype,
                const HTShape& shape, int src_device_index = -1,
                const DeviceGroup& device_group = DeviceGroup())
  : OpInterface(quote(P2PRecvOp)), _src_group(src_group), _dtype(dtype),
                _shape(shape), _src_device_index(src_device_index) {
    HT_ASSERT(!src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(device_group.empty() ||
              src_group.num_devices() == device_group.num_devices())
      << "Currently we require equal tensor parallelism degree across "
      << "P2P communication. Got " << src_group << " vs. " << device_group;
    HT_ASSERT(!shape.empty())
      << "P2P RecvOp require determined tensor shape to recv. Got empty shape param!";
    HT_ASSERT(dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_RECV_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  const DeviceGroup& src_group() const {
    return _src_group;
  }

  int src_device_index() {
    return _src_device_index;
  } 

 protected:
  DeviceGroup _src_group;
  int _src_device_index{-1};
  DataType _dtype;
  HTShape _shape;           
};

Tensor MakeP2PRecvOp(const DeviceGroup& src_group, DataType dtype,
                     const HTShape& shape, int src_device_index = -1, 
                     OpMeta op_meta = OpMeta());

class BatchedISendIRecvOpImpl final : public OpInterface {
 public:
  BatchedISendIRecvOpImpl(const std::vector<Device>& dst_devices, 
                          const HTShapeList& outputs_shape,
                          const std::vector<Device>& src_devices, 
                          const std::vector<Device>& comm_devices,
                          DataType dtype)
  : OpInterface(quote(BatchedISendIRecvOp)), _dst_devices(dst_devices), 
  _outputs_shape(outputs_shape), _src_devices(src_devices), 
  _comm_devices(comm_devices), _dtype(dtype) {}

  uint64_t op_indicator() const noexcept override {
    return BATCHED_ISEND_IRECV_OP;
  }

 public:
  void print_mesg(Operator& op) {
    std::ostringstream os;
    os << "dst devices =";
    for (auto& d : _dst_devices) {
      os << " device_" << d.index();
    }
    os << "src devices =";
    for (auto& s : _src_devices) {
      os << " device_" << s.index();
    }
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() 
                 << ": BatchedISendIRecvOp definition: " << op->name() << ": " << os.str();    
  }

  const std::vector<Device>& src_devices() const {
    return _src_devices;
  }

  std::vector<Device>& src_devices() {
    return _src_devices;
  }  

  const std::vector<Device>& dst_devices() const {
    return _dst_devices;
  }

  std::vector<Device>& dst_devices() {
    return _dst_devices;
  }

 protected:
  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override; 

  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;   

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  std::vector<Device> _dst_devices; 
  std::vector<Device> _src_devices;
  std::vector<Device> _comm_devices;
  HTShapeList _outputs_shape;
  DataType _dtype;
};

Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               const std::vector<Device>& dst_devices, 
                               const HTShapeList& outputs_shape, 
                               const std::vector<Device>& src_devices, 
                               const std::vector<Device>& comm_devices, 
                               DataType dtype, OpMeta op_meta = OpMeta());

class AllGatherOpImpl final : public OpInterface {
 public:
  AllGatherOpImpl(const DeviceGroup& comm_group, const DeviceGroup& device_group = DeviceGroup())
  : OpInterface(quote(AllGatherOp)), _comm_group(comm_group) {
    HT_ASSERT(comm_group.num_devices() >= 2)
      << "AllGather requires two or more devices. Got " << comm_group;
    if (!device_group.empty()) {
      for (int i = 0; i < comm_group.num_devices(); i++) {
        HT_ASSERT(device_group.contains(comm_group.get(i)))
          << "AllGather: device in comm_group: " << comm_group.get(i) 
          << " must in device group: " << device_group;
      }
    }
  }

  uint64_t op_indicator() const noexcept override {
    return ALL_GATHER_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  DeviceGroup _comm_group;
};

Tensor MakeAllGatherOp(Tensor input, const DeviceGroup& comm_group,
                       OpMeta op_meta = OpMeta());

class ReduceScatterOpImpl final : public OpInterface {
 public:
  ReduceScatterOpImpl(const DeviceGroup& comm_group, const DeviceGroup& device_group = DeviceGroup())
  : OpInterface(quote(ReduceScatterOp)), _comm_group(comm_group) {
    HT_ASSERT(comm_group.num_devices() >= 2)
      << "ReduceScatter requires two or more devices. Got " << comm_group;          
    if (!device_group.empty()) {
      for (int i = 0; i < comm_group.num_devices(); i++) {
        HT_ASSERT(device_group.contains(comm_group.get(i)))
          << "ReduceScatter: device in comm_group: " << comm_group.get(i) 
          << " must in device group: " << device_group;
      }
    }    
  }

  uint64_t op_indicator() const noexcept override {
    return REDUCE_SCATTER_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                                                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  DeviceGroup _comm_group;
};

Tensor MakeReduceScatterOp(Tensor input, const DeviceGroup& comm_group,
                           OpMeta op_meta = OpMeta());

}
}