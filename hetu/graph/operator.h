#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/utils/context_store.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace graph {

class OpMeta {
 public:
  OpMeta() = default;

  OpMeta(const OpMeta&) = default;
  OpMeta(OpMeta&&) = default;
  OpMeta& operator=(OpMeta&&) = default;
  OpMeta& operator=(const OpMeta&) = default;

  inline OpMeta& set_name(const OpName& n) {
    name = n;
    return *this;
  }

  inline OpMeta& set_name(OpName&& n) {
    name = std::move(n);
    return *this;
  }

  inline OpMeta& set_stream_index(StreamIndex si) {
    // TODO: check whether the stream index is valid
    stream_index = si;
    return *this;
  }

  inline OpMeta& set_eager_device(const Device& device) {
    eager_device = device;
    return *this;
  }

  inline OpMeta& set_eager_device(Device&& device) {
    eager_device = std::move(device);
    return *this;
  }

  inline OpMeta& set_device_group(const DeviceGroup& group) {
    device_group = group;
    return *this;
  }

  inline OpMeta& set_device_group(DeviceGroup&& group) {
    device_group = std::move(group);
    return *this;
  }

  inline OpMeta& set_extra_deps(const TensorList& deps) {
    extra_deps = deps;
    return *this;
  }

  inline OpMeta& set_extra_deps(TensorList&& deps) {
    extra_deps = std::move(deps);
    return *this;
  }

  inline OpMeta& set_is_deduce_states(bool deduce_states) {
    is_deduce_states = deduce_states;
    return *this;
  }
  
  inline OpMeta& set(const OpMeta& other) {
    operator=(other);
    return *this;
  }

  inline OpMeta& set(OpMeta&& other) {
    operator=(std::move(other));
    return *this;
  }

  static OpMeta Merge(const OpMeta& base_meta, const OpMeta& new_meta) {
    OpMeta ret = base_meta;
    if (!new_meta.name.empty())
      ret.set_name(new_meta.name);
    if (new_meta.stream_index != kUndeterminedStream)
      ret.set_stream_index(new_meta.stream_index);
    if (!new_meta.eager_device.is_undetermined())
      ret.set_eager_device(new_meta.eager_device);
    if (!new_meta.device_group.empty())
      ret.set_device_group(new_meta.device_group);
    if (!new_meta.extra_deps.empty())
      ret.set_extra_deps(new_meta.extra_deps);
    return ret;
  }

  OpName name;
  StreamIndex stream_index{kUndeterminedStream};
  Device eager_device{kUndeterminedDevice};
  DeviceGroup device_group;
  TensorList extra_deps;
  bool is_deduce_states{true};  
};

std::ostream& operator<<(std::ostream&, const OpMeta&);

using OpRuntimeContext = ContextStore;

class RuntimeContext {
 public:
  RuntimeContext(size_t init_capacity) {
    _ctxs.reserve(init_capacity);
  }

  ~RuntimeContext() {
    for (auto& kv : _ctxs)
      delete kv.second;
    _ctxs.clear();
  }

  OpRuntimeContext& get_or_create(OpId id) {
    auto it = _ctxs.find(id);
    if (it != _ctxs.end()) {
      return *it->second;
    } else {
      auto* op_ctx = new OpRuntimeContext();
      _ctxs[id] = op_ctx;
      return *op_ctx;
    }
  }

  OpRuntimeContext& get(OpId id) {
    return *_ctxs.at(id);
  }

  const OpRuntimeContext& get(OpId id) const {
    return *_ctxs.at(id);
  }

  void remove(OpId id) {
    _ctxs.erase(id);
  }

  void clear() {
    _ctxs.clear();
  }

 private:
  std::unordered_map<OpId, OpRuntimeContext*> _ctxs;
};

struct OpInstantiationContext {
  DeviceGroup placement_group;
  Device placement;
  StreamIndex stream_index;
  std::unique_ptr<Event> start[HT_MAX_NUM_MICRO_BATCHES];
  std::unique_ptr<Event> stop[HT_MAX_NUM_MICRO_BATCHES];

  Stream stream() const {
    // Question: create stream inside kernels?
    return Stream(placement, stream_index);
  }
};

class OpInterface : public shared_ptr_target {
 protected:
  friend class OpDef;
  friend class Operator;
  OpInterface(OpType&& op_type) : _type(op_type) {}

 public:
  ~OpInterface() = default;

  // disable copy constructor and move constructor
  OpInterface(const OpInterface&) = delete;
  OpInterface& operator=(const OpInterface&) = delete;
  OpInterface(OpInterface&&) = delete;
  OpInterface& operator=(OpInterface&&) = delete;

  inline const OpType& type() const {
    return _type;
  }

  virtual uint64_t op_indicator() const noexcept {
    return 0;
  }

  virtual bool operator==(const OpInterface& rhs) const {
    return op_indicator() == rhs.op_indicator() && type() == rhs.type();
  }

  bool operator!=(const OpInterface& rhs) const {
    return !(*this == rhs);
  }

  virtual bool inplace_at(size_t input_position) const {
    return false;
  }

  inline std::vector<NDArrayMeta> InferMeta(const TensorList& inputs) const {
    return DoInferMeta(inputs);
  }

  inline void DeduceStates(const TensorList& inputs, TensorList& outputs, 
                           const OpMeta& op_meta) const {
    return DoDeduceStates(inputs, outputs, op_meta);
  }

  inline TensorList Gradient(Operator& op,
                             const TensorList& grad_outputs) const {
    return DoGradient(op, grad_outputs);
  }

  inline bool MapToParallelDevices(Operator& op,
                                   const DeviceGroup& placement_group) const {
    return DoMapToParallelDevices(op, placement_group);
  }

  inline bool Instantiate(Operator& op, const Device& placement,
                          StreamIndex stream_id) const {
    return DoInstantiate(op, placement, stream_id);
  }

  inline HTShapeList InferShape(Operator& op, const HTShapeList& shapes,
                                RuntimeContext& runtime_ctx) const {
    return DoInferShape(op, shapes, runtime_ctx);
  }

  inline NDArrayList AllocOutputs(Operator& op, const NDArrayList& inputs,
                                  RuntimeContext& runtime_ctx) const {
    return DoAllocOutputs(op, inputs, runtime_ctx);
  }

  inline void Compute(Operator& op, const NDArrayList& inputs,
                      NDArrayList& outputs, RuntimeContext& runtime_ctx) const {
    DoCompute(op, inputs, outputs, runtime_ctx);
  }

  inline NDArrayList Compute(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const {
    return DoCompute(op, inputs, runtime_ctx);
  }

 protected:
  virtual std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const = 0;

  virtual void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                              const OpMeta& op_meta) const;

  virtual TensorList DoGradient(Operator&, const TensorList&) const {
    HT_RUNTIME_ERROR << "Op with type " << type() << "is not differentiable";
    __builtin_unreachable();
  }

  virtual bool DoMapToParallelDevices(Operator& op,
                                      const DeviceGroup& placement_group) const;

  virtual bool DoInstantiate(Operator& op, const Device& placement,
                             StreamIndex stream_id) const;

  virtual HTShapeList DoInferShape(Operator& op,
                                   const HTShapeList& input_shapes,
                                   RuntimeContext& runtime_ctx) const;

  virtual NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                                     RuntimeContext& runtime_ctx) const;

  virtual void DoCompute(Operator& op, const NDArrayList& inputs,
                         NDArrayList& outputs,
                         RuntimeContext& runtime_ctx) const = 0;

  virtual NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                                RuntimeContext& runtime_ctx) const {
    auto outputs = DoAllocOutputs(op, inputs, runtime_ctx);
    // NDArrayList f32inputs = inputs;
    // for (auto& input: f32inputs) {
    //   if (input->dtype() != DataType::FLOAT32)
    //     input = NDArray::to(input, input->device(), DataType::FLOAT32);
    //   else
    //     input = NDArray::copy(input);
    // }
    // auto f32outputs = DoAllocOutputs(op, f32inputs, runtime_ctx);
    // for (auto& output: f32outputs) {
    //   if (output->dtype() != DataType::FLOAT32)
    //     output = NDArray::to(output, output->device(), DataType::FLOAT32);
    //   else
    //     output = NDArray::copy(output);
    // }
    DoCompute(op, inputs, outputs, runtime_ctx);
    // if (_type != "DataTransfer") {
    //   HT_LOG_INFO << _type <<  "\nInputs";
    //   for (auto& input: inputs){
    //     HT_LOG_INFO << input->raw_data_ptr() <<  " "  << NDArray::sum(NDArray::abs(input, kBlockingStream), {0}, false, kBlockingStream) << " " << input;
    //   }
    //   HT_LOG_INFO << "\nOutputs";
    //   for (auto& output: outputs){
    //     HT_LOG_INFO << output->raw_data_ptr() <<  " "  << NDArray::sum(NDArray::abs(output, kBlockingStream), {0}, false, kBlockingStream) << " " << output;
    //   }
    //   // HT_LOG_INFO  << "\nF32Inputs:" << f32inputs << "\nF32Outputs:" << f32outputs;
    // }
    return outputs;
  }

  const OpType _type;
};

struct OpIdentifier {
  GraphId graph_id;
  OpId op_id;
};

class OpDef : public shared_ptr_target {
 protected:
  friend class Operator;
  friend class Graph;
  friend class DefineByRunGraph;
  friend class DefineAndRunGraph;
  friend class ExecutableGraph;
  struct constrcutor_access_key {};

 public:
  OpDef(const constrcutor_access_key&, OpIdentifier ids,
        std::shared_ptr<OpInterface> body, TensorList inputs, OpMeta op_meta);

  ~OpDef() = default;

  // disable copy constructor and move constructor
  OpDef(const OpDef&) = delete;
  OpDef& operator=(const OpDef&) = delete;
  OpDef(OpDef&&) = delete;
  OpDef& operator=(OpDef&&) = delete;

  inline TensorList Gradient(const TensorList& grad_outputs) {
    return _body->Gradient(get_self(), grad_outputs);
  }

  inline bool MapToParallelDevices(const DeviceGroup& placement_group) {
    return _body->MapToParallelDevices(get_self(), placement_group);
  }

  inline void DeduceStates() {
    return _body->DeduceStates(inputs(), outputs(), op_meta());
  }

  inline bool Instantiate(const Device& placement, StreamIndex stream_id) {
    return _body->Instantiate(get_self(), placement, stream_id);
  }

  HTShapeList InferShape(const HTShapeList& input_shapes,
                         RuntimeContext& runtime_ctx) {
    return _body->InferShape(get_self(), input_shapes, runtime_ctx);
  }

<<<<<<< HEAD
  NDArrayList Compute(const NDArrayList& inputs, RuntimeContext& runtime_ctx) {
    BlockOrSyncAllInputs();
    instantiation_ctx().start->Record(stream());
    // HT_LOG_INFO << name() << "\n" << inputs;
    auto ret = _body->Compute(get_self(), inputs, runtime_ctx);
    // if (ret.size() > 0)
    // HT_LOG_INFO << ret[0];
    instantiation_ctx().stop->Record(stream());
=======
  NDArrayList Compute(const NDArrayList& inputs, RuntimeContext& runtime_ctx, size_t micro_batch_id = 0) {
    HT_ASSERT(micro_batch_id < HT_MAX_NUM_MICRO_BATCHES)
      << "Num micro batches muse <= " << HT_MAX_NUM_MICRO_BATCHES 
      << ", got micro batch id: " << micro_batch_id;
    BlockOrSyncAllInputs(micro_batch_id);
    instantiation_ctx().start[micro_batch_id]->Record(stream());
    auto ret = _body->Compute(get_self(), inputs, runtime_ctx);
    instantiation_ctx().stop[micro_batch_id]->Record(stream());
>>>>>>> 984218bd9d37ecf54e6ecb64b54a53a9f283b8fe
    return ret;
  }

  void Sync(size_t micro_batch_id = 0) {
    instantiation_ctx().stop[micro_batch_id]->Sync();
  }

  OpId id() const noexcept {
    return _ids.op_id;
  }

  GraphId graph_id() const noexcept {
    return _ids.graph_id;
  }

  const OpType& type() const noexcept {
    return _body->type();
  }

  const OpInterface& body() const {
    return *_body;
  }

  // for op interface specific func call
  OpInterface& body() {
    return *_body;
  }  

  const OpName& name() const noexcept {
    return _op_meta.name;
  }

  OpName grad_name(size_t input_id = 0) const {
    return name() + "_grad_" + input(input_id)->name();
  }

  const Device& eager_device() const noexcept {
    return _op_meta.eager_device;
  }

  const DeviceGroup& device_group() const noexcept {
    return _op_meta.device_group;
  }

  const OpMeta& op_meta() const noexcept {
    return _op_meta;
  }

  OpMeta grad_op_meta() const {
    return OpMeta()
      .set_stream_index(stream_index())
      .set_device_group(device_group());
  }

  bool inplace_at(size_t input_position) const {
    return _body->inplace_at(input_position);
  }

  const TensorList& inputs() const noexcept {
    return _inputs;
  }

  TensorList& inputs() noexcept {
    return _inputs;
  }

  const TensorList& outputs() const noexcept {
    return _outputs;
  }

  TensorList& outputs() noexcept {
    return _outputs;
  }

  const Tensor& input(size_t i) const {
    return _inputs[i];
  }

  Tensor& input(size_t i) {
    return _inputs[i];
  }

  const Tensor& output(size_t i) const {
    return _outputs[i];
  }

  Tensor& output(size_t i) {
    return _outputs[i];
  }

  size_t num_inputs() const {
    return _inputs.size();
  }

  size_t num_outputs() const {
    return _outputs.size();
  }

  const TensorList& in_dep_linkers() const noexcept {
    return _extra_in_dep_linkers;
  }

  TensorList& in_dep_linkers() noexcept {
    return _extra_in_dep_linkers;
  }

  const Tensor& in_dep_linker(size_t i) const {
    return _extra_in_dep_linkers[i];
  }

  Tensor& in_dep_linker(size_t i) {
    return _extra_in_dep_linkers[i];
  }

  size_t num_in_dep_linkers() const {
    return _extra_in_dep_linkers.size();
  }

  const Tensor& out_dep_linker() const noexcept {
    return _extra_out_dep_linkers.front();
  }

  Tensor& out_dep_linker() noexcept {
    return _extra_out_dep_linkers.front();
  }

  size_t in_degrees() const {
    return num_inputs() + _extra_in_dep_linkers.size();
  }

  uint64_t op_indicator() const noexcept {
    return _body->op_indicator();
  }

  const DeviceGroup& placement_group() const noexcept {
    return _inst_ctx.placement_group;
  }

  const Device& placement() const noexcept {
    return _inst_ctx.placement;
  }

  Stream stream() const noexcept {
    return Stream(placement(), stream_index());
  }

  StreamIndex stream_index() const noexcept {
    return _inst_ctx.stream_index;
  }

  const OpInstantiationContext& instantiation_ctx() const {
    return _inst_ctx;
  }

  OpInstantiationContext& instantiation_ctx() {
    return _inst_ctx;
  }

  bool is_computed() const {
    return false;
  }

  bool is_parameter() const;

  bool requires_grad(size_t i) const {
    return _inputs[i]->requires_grad();
  }

 protected:
  // Walkaround methods to get the corresponding wrapper
  Operator& get_self();

  const Operator& get_self() const;

  void BlockOrSyncAllInputs(size_t micro_batch_id = 0);
  
  void BlockOrSyncInput(Tensor& input, size_t micro_batch_id = 0);

  const OpIdentifier _ids;
  std::shared_ptr<OpInterface> _body;

  TensorList _inputs;
  TensorList _outputs;
  TensorList _extra_in_dep_linkers;
  TensorList _extra_out_dep_linkers;

  OpMeta _op_meta;
  OpInstantiationContext _inst_ctx;
};

class Operator : public shared_ptr_wrapper<OpDef> {
 protected:
  friend class Graph;
  friend class DefineByRunGraph;
  friend class DefineAndRunGraph;
  friend class ExecutableGraph;

  Operator(OpIdentifier ids, std::shared_ptr<OpInterface> body,
           TensorList inputs, OpMeta op_meta = OpMeta());

 public:
  Operator() = default;

  /******************************************************
   * Helper functions
   ******************************************************/
 public:
  template <typename UnaryFunction>
  static void for_each_input_tensor(Operator& op, UnaryFunction fn) {
    for (auto& tensor : op->_inputs)
      fn(tensor);
    for (auto& tensor : op->_extra_in_dep_linkers)
      fn(tensor);
  }

  template <typename UnaryFunction>
  static void for_each_input_tensor(const Operator& op, UnaryFunction fn) {
    for (const auto& tensor : op->_inputs)
      fn(tensor);
    for (const auto& tensor : op->_extra_in_dep_linkers)
      fn(tensor);
  }

  template <typename UnaryFunction>
  static auto transform_each_input_tensor(Operator& op, UnaryFunction fn) {
    std::vector<decltype(fn(op->_inputs.front()))> transformed_inputs;
    std::vector<decltype(fn(op->_inputs.front()))> transformed_in_deps;
    transformed_inputs.reserve(op->_inputs.size());
    transformed_in_deps.reserve(op->_extra_in_dep_linkers.size());
    for (auto& tensor : op->_inputs)
      transformed_inputs.push_back(fn(tensor));
    for (auto& tensor : op->_extra_in_dep_linkers)
      transformed_in_deps.push_back(fn(tensor));
    return std::make_tuple(transformed_inputs, transformed_in_deps);
  }

  template <typename UnaryFunction>
  static auto transform_each_input_tensor(const Operator& op,
                                          UnaryFunction fn) {
    std::vector<decltype(fn(op->_inputs.front()))> transformed_inputs;
    std::vector<decltype(fn(op->_inputs.front()))> transformed_in_deps;
    transformed_inputs.reserve(op->_inputs.size());
    transformed_in_deps.reserve(op->_extra_in_dep_linkers.size());
    for (const auto& tensor : op->_inputs)
      transformed_inputs.push_back(fn(tensor));
    for (const auto& tensor : op->_extra_in_dep_linkers)
      transformed_in_deps.push_back(fn(tensor));
    return std::make_tuple(transformed_inputs, transformed_in_deps);
  }

  template <typename BinaryFunction>
  static void for_each_input_tensor_pair(Operator& op1, Operator& op2,
                                         BinaryFunction fn) {
    for (size_t i = 0; i < op1->_inputs.size(); i++)
      fn(op1->_inputs.at(i), op2->_inputs.at(i));
    for (size_t i = 0; i < op1->_extra_in_dep_linkers.size(); i++)
      fn(op1->_extra_in_dep_linkers.at(i), op2->_extra_in_dep_linkers.at(i));
  }

  template <typename BinaryFunction>
  static void for_each_input_tensor_pair(const Operator& op1,
                                         const Operator& op2,
                                         BinaryFunction fn) {
    for (size_t i = 0; i < op1->_inputs.size(); i++)
      fn(op1->_inputs.at(i), op2->_inputs.at(i));
    for (size_t i = 0; i < op1->_extra_in_dep_linkers.size(); i++)
      fn(op1->_extra_in_dep_linkers.at(i), op2->_extra_in_dep_linkers.at(i));
  }

  template <typename UnaryFunction>
  static void for_each_output_tensor(Operator& op, UnaryFunction fn) {
    if (op->_outputs.empty()) {
      fn(op->_extra_out_dep_linkers.front());
    } else {
      for (auto& tensor : op->_outputs)
        fn(tensor);
    }
  }

  template <typename UnaryFunction>
  static void for_each_output_tensor(const Operator& op, UnaryFunction fn) {
    if (op->_outputs.empty()) {
      fn(op->_extra_out_dep_linkers.front());
    } else {
      for (const auto& tensor : op->_outputs)
        fn(tensor);
    }
  }

  template <typename BinaryFunction>
  static void for_each_output_tensor_pair(Operator& op1, Operator& op2,
                                          BinaryFunction fn) {
    if (op1->_outputs.empty()) {
      fn(op1->_extra_out_dep_linkers.front(),
         op2->_extra_out_dep_linkers.front());
    } else {
      for (size_t i = 0; i < op1->_outputs.size(); i++)
        fn(op1->_outputs.at(i), op2->_outputs.at(i));
    }
  }

  template <typename BinaryFunction>
  static void for_each_output_tensor_pair(const Operator& op1,
                                          const Operator& op2,
                                          BinaryFunction fn) {
    if (op1->_outputs.empty()) {
      fn(op1->_extra_out_dep_linkers.front(), op2->_extra_out_dep_linkers.front());
    } else {
      for (size_t i = 0; i < op1->_outputs.size(); i++)
        fn(op1->_outputs.at(i), op2->_outputs.at(i));
    }
  }

  template <typename UnaryPredicate>
  static bool all_input_tensors_of(Operator& op, UnaryPredicate pred) {
    for (auto& tensor : op->_inputs)
      if (!pred(tensor))
        return false;
    for (auto& tensor : op->_extra_in_dep_linkers)
      if (!pred(tensor))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_input_tensors_of(const Operator& op, UnaryPredicate pred) {
    for (const auto& tensor : op->_inputs)
      if (!pred(tensor))
        return false;
    for (const auto& tensor : op->_extra_in_dep_linkers)
      if (!pred(tensor))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_output_tensors_of(Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (auto& tensor : op->_outputs)
        if (!pred(tensor))
          return false;
    }
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_output_tensors_of(const Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (const auto& tensor : op->_outputs)
        if (!pred(tensor))
          return false;
    }
    return true;
  }
};

std::ostream& operator<<(std::ostream&, const Operator&);

/******************************************************
 * Indicators of Operators
 ******************************************************/

static const uint64_t PLACEHOLDER_OP = 1ul << 1;
static const uint64_t VARIABLE_OP = 1ul << 2;
static const uint64_t HOST_TO_DEVICE_OP = 1ul << 3;
static const uint64_t DEVICE_TO_HOST_OP = 1ul << 4;
static const uint64_t PEER_TO_PEER_SEND_OP = 1ul << 5;
static const uint64_t PEER_TO_PEER_RECV_OP = 1ul << 6;
static const uint64_t ALL_TO_ALL_OP = 1ul << 7;
static const uint64_t ALL_REDUCE_OP = 1ul << 8;
static const uint64_t ALL_GATHER_OP = 1ul << 9;
static const uint64_t REDUCE_SCATTER_OP = 1ul << 10;
static const uint64_t BROADCAST_OP = 1ul << 11;
static const uint64_t REDUCE_OP = 1ul << 12;
static const uint64_t P2P_OP = 1ul << 13;
static const uint64_t BATCHED_ISEND_IRECV_OP = 1ul << 14;
static const uint64_t GATHER_OP = 1ul << 15;
static const uint64_t SCATTER_OP = 1ul << 16;
static const uint64_t COMM_SPLIT_OP = 1ul << 19;
static const uint64_t COMM_OP = 1ul << 20;
static const uint64_t UNKNOWN_OP = 1ul << 21;
static const uint64_t SLICE_OP = 1ul << 59;
static const uint64_t LOSS_OP = 1ul << 60;
static const uint64_t LOSS_GRADIENT_OP = 1ul << 61;
static const uint64_t OPTIMIZER_UPDATE_OP = 1ul << 62;
static const uint64_t GROUP_OP = 1ul << 63;

#define DECLARE_OP_INDICATOR_CHECKER(type, indicator)                          \
  inline bool is_##type##_op(const OpInterface& x) {                           \
    return (x.op_indicator() & (indicator)) != 0;                              \
  }                                                                            \
  inline bool is_##type##_op(const Operator& x) {                              \
    return is_##type##_op(x->body());                                          \
  }                                                                            \
  inline bool is_##type##_op(const OpRef& x) {                                 \
    return is_##type##_op(x.get());                                            \
  }                                                                            \
  inline bool is_##type##_op(const OpCRef& x) {                                \
    return is_##type##_op(x.get());                                            \
  }

DECLARE_OP_INDICATOR_CHECKER(placeholder, PLACEHOLDER_OP)
DECLARE_OP_INDICATOR_CHECKER(variable, VARIABLE_OP)
DECLARE_OP_INDICATOR_CHECKER(host_to_device, HOST_TO_DEVICE_OP)
DECLARE_OP_INDICATOR_CHECKER(device_to_host, DEVICE_TO_HOST_OP)
DECLARE_OP_INDICATOR_CHECKER(peer_to_peer_send, PEER_TO_PEER_SEND_OP)
DECLARE_OP_INDICATOR_CHECKER(peer_to_peer_recv, PEER_TO_PEER_RECV_OP)
DECLARE_OP_INDICATOR_CHECKER(all_to_all, ALL_TO_ALL_OP)
DECLARE_OP_INDICATOR_CHECKER(all_reduce, ALL_REDUCE_OP)
DECLARE_OP_INDICATOR_CHECKER(all_gather, ALL_GATHER_OP)
DECLARE_OP_INDICATOR_CHECKER(reduce_scatter, REDUCE_SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(broadcast, BROADCAST_OP)
DECLARE_OP_INDICATOR_CHECKER(reduce, REDUCE_OP)
DECLARE_OP_INDICATOR_CHECKER(p2p, P2P_OP)
DECLARE_OP_INDICATOR_CHECKER(batched_isend_irecv, BATCHED_ISEND_IRECV_OP)
DECLARE_OP_INDICATOR_CHECKER(gather, GATHER_OP)
DECLARE_OP_INDICATOR_CHECKER(scatter, SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(comm_split, COMM_SPLIT_OP)
DECLARE_OP_INDICATOR_CHECKER(comm, COMM_OP)
DECLARE_OP_INDICATOR_CHECKER(unknown, UNKNOWN_OP)
DECLARE_OP_INDICATOR_CHECKER(communucation,
                             PEER_TO_PEER_SEND_OP | PEER_TO_PEER_RECV_OP |
                               ALL_TO_ALL_OP | ALL_REDUCE_OP |
                               ALL_GATHER_OP | REDUCE_SCATTER_OP |
                               BROADCAST_OP | REDUCE_OP |
                               P2P_OP | BATCHED_ISEND_IRECV_OP |
                               GATHER_OP | SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(slice, SLICE_OP)
DECLARE_OP_INDICATOR_CHECKER(loss, LOSS_OP)
DECLARE_OP_INDICATOR_CHECKER(loss_gradient, LOSS_GRADIENT_OP)
DECLARE_OP_INDICATOR_CHECKER(optimizer_update, OPTIMIZER_UPDATE_OP)
DECLARE_OP_INDICATOR_CHECKER(group, GROUP_OP)

inline StreamIndex get_suggested_stream_index(const Operator& op) {
  if (is_host_to_device_op(op)) {
    return kH2DStream;
  } else if (is_device_to_host_op(op)) {
    return kD2HStream;
  } else if (is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op)) {
    return kP2PStream;
  } else if (is_all_to_all_op(op)) {
    return kCollectiveStream;
  } else {
    return kComputingStream;
  }
}

} // namespace graph
} // namespace hetu
