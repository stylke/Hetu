#include "hetu/graph/headers.h"
#include "hetu/graph/ops/group.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {
namespace graph {

bool OpInterface::DoMapToParallelDevices(Operator& op,
                                         const DeviceGroup& pg) const {
  op->instantiation_ctx().placement_group = pg;
  // TODO: set output statuses
  return true;
}

bool OpInterface::DoInstantiate(Operator& op, const Device& placement,
                                StreamIndex stream_index) const {
  auto& inst_ctx = op->instantiation_ctx();
  inst_ctx.placement = placement;
  inst_ctx.stream_index = stream_index;
  if (placement.is_cuda()) {
    inst_ctx.start = std::make_unique<hetu::impl::CUDAEvent>(placement);
    inst_ctx.stop = std::make_unique<hetu::impl::CUDAEvent>(placement);
  } else {
    inst_ctx.start = std::make_unique<hetu::impl::CPUEvent>();
    inst_ctx.stop = std::make_unique<hetu::impl::CPUEvent>();
  }
  Operator::for_each_output_tensor(
    op, [&](Tensor& tensor) { tensor->set_placement(placement); });
  return true;
}

HTShapeList OpInterface::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& runtime_ctx) const {
  if (op->num_outputs() == 0)
    return HTShapeList();
  HT_NOT_IMPLEMENTED << "InferShape fn of op \"" << op->type()
                     << "\" is not defined";
  __builtin_unreachable();
}

NDArrayList OpInterface::DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                                        RuntimeContext& runtime_ctx) const {
  NDArrayList outputs;
  if (op->num_outputs() > 0) {
    outputs.reserve(op->num_outputs());
    HTShapeList input_shapes;
    input_shapes.reserve(op->num_inputs());
    for (auto& input : inputs)
      input_shapes.push_back(input->shape());
    // Although we have inferred the meta of tensors,
    // InferShape is still necessary in pipeline parallelism
    auto output_shapes = DoInferShape(op, input_shapes, runtime_ctx);
    for (size_t i = 0; i < output_shapes.size(); i++) {
      outputs.push_back(NDArray::empty(output_shapes[i],
                                       op->instantiation_ctx().placement,
                                       op->output(i)->dtype()));
    }
  }
  return outputs;
}

OpDef::OpDef(const constrcutor_access_key&, OpIdentifier ids,
             std::shared_ptr<OpInterface> body, TensorList inputs,
             OpMeta op_meta)
: _ids{std::move(ids)},
  _body(std::move(body)),
  _inputs(std::move(inputs)),
  _op_meta(std::move(op_meta)) {
  auto& graph = Graph::GetGraph(_ids.graph_id);
  // Question: Is op name really necessary?
  if (_op_meta.name.empty()) {
    auto cnt = graph.get_op_type_cnt(_body->type());
    if (cnt == 0)
      _op_meta.name = _body->type();
    else
      _op_meta.name = _body->type() + '(' + std::to_string(cnt) + ')';
  }
  // All inputs must be tensors
  for (size_t i = 0; i < _inputs.size(); i++) {
    HT_VALUE_ERROR_IF((!_inputs[i].is_defined()) ||
                      _inputs[i]->is_out_dep_linker())
      << "Failed to construct the \"" << _body->type() << "\" op: "
      << "Cannot convert input " << i << " to a tensor: " << _inputs[i] << ".";
  }
  // Extra input depencenies. May be tensors or output dependency linkers
  auto& extra_deps = _op_meta.extra_deps;
  if (extra_deps.size() <= 1 || is_group_op(*_body)) {
    // Walkaround: if we are constructing a group op,
    // we shall not construct another group op to handle the dependecies.
    _extra_in_dep_linkers = extra_deps;
  } else {
    // Merge dependencies into a group op
    _extra_in_dep_linkers.push_back(
      MakeGroupOp(OpMeta()
                    .set_extra_deps(extra_deps)
                    .set_name(_op_meta.name + "_extra_deps")));
  }
  // Outputs of this op
  auto output_meta_list = _body->InferMeta(_inputs);
  if (output_meta_list.size() == 1) {
    auto& output_meta = output_meta_list.front();
    HT_ASSERT(output_meta.dtype != kUndeterminedDataType)
      << "Data type is not provided for output " << _outputs.size()
      << " of the \"" << _body->type() << "\" op.";
    _outputs.emplace_back(
      TensorIdentifier{_ids.graph_id, _ids.op_id, 0, graph.next_tensor_id()},
      _op_meta.name, output_meta);
  } else if (output_meta_list.size() > 1) {
    _outputs.reserve(output_meta_list.size());
    for (int i = 0; i < static_cast<int>(output_meta_list.size()); i++) {
      // (TensorIdentifier ids, TensorName name, NDArrayMeta meta)
      auto& output_meta = output_meta_list[i];
      HT_ASSERT(output_meta.dtype != kUndeterminedDataType)
        << "Data type is not provided for output " << i << " of the \""
        << _body->type() << "\" op.";
      _outputs.emplace_back(
        TensorIdentifier{_ids.graph_id, _ids.op_id, i, graph.next_tensor_id()},
        _op_meta.name + '_' + std::to_string(i), output_meta);
    }
  } else {
    _extra_out_dep_linkers.emplace_back(
      TensorIdentifier{_ids.graph_id, _ids.op_id, -1, graph.next_tensor_id()},
      _op_meta.name);
  }
}

Operator& OpDef::get_self() {
  return Graph::GetGraph(graph_id()).GetOp(id());
}

const Operator& OpDef::get_self() const {
  return Graph::GetGraph(graph_id()).GetOp(id());
}

void OpDef::BlockOrSyncAllInputs() {
  for (auto& input : _inputs)
    BlockOrSyncInput(input);
  for (auto& in_dep : _extra_in_dep_linkers)
    BlockOrSyncInput(in_dep);
}

void OpDef::BlockOrSyncInput(Tensor& input) {
  if (!input.is_defined())
    return;
  auto& input_op = input->producer();
  const auto& input_placement = input_op->instantiation_ctx().placement;
  const auto& current_placement = instantiation_ctx().placement;
  HT_RUNTIME_ERROR_IF(input_placement.is_undetermined() ||
                      (!input_placement.local()))
    << "Input " << input << " is not instantiated or on a remote device. "
    << "Please use P2P communication op to fetch it";
  if (input_placement != current_placement) {
    // We cannot block different devices. Just sync here.
    input_op->instantiation_ctx().stop->Sync();
  } else if (input_op->instantiation_ctx().stream_index !=
             instantiation_ctx().stream_index) {
    // Both ops are on the same device. We can block the current op
    // by waiting for the stop event of the dependency.
    input_op->instantiation_ctx().stop->Block(instantiation_ctx().stream());
  }
}

Operator::Operator(OpIdentifier ids, std::shared_ptr<OpInterface> body,
                   TensorList inputs, OpMeta op_meta) {
  this->_ptr =
    make_ptr<OpDef>(OpDef::constrcutor_access_key(), std::move(ids),
                    std::move(body), std::move(inputs), std::move(op_meta));
}

std::ostream& operator<<(std::ostream& os, const OpMeta& meta) {
  os << "{";
  bool first = true;
  if (meta.stream_index != kUndeterminedStream) {
    if (!first)
      os << ", ";
    os << "stream_index=" << meta.stream_index;
    first = false;
  }
  if (!meta.eager_device.is_undetermined()) {
    if (!first)
      os << ", ";
    os << "eager_device=" << meta.eager_device;
    first = false;
  }
  if (!meta.device_group.empty()) {
    if (!first)
      os << ", ";
    os << "device_group=" << meta.device_group;
    first = false;
  }
  if (!meta.extra_deps.empty()) {
    if (!first)
      os << ", ";
    os << "extra_deps=" << meta.extra_deps;
    first = false;
  }
  if (!meta.name.empty()) {
    if (!first)
      os << ", ";
    os << "name=" << meta.name;
    first = false;
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  if (op.is_defined())
    os << op->name();
  else
    os << "Operator()";
  return os;
}

} // namespace graph
} // namespace hetu
