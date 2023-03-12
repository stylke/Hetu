#include "hetu/graph/eager_graph.h"

namespace hetu {
namespace graph {

Operator& EagerGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                  TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  auto& op =
    MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));

  // Eager instantiation and execution
  Device placement = op->eager_device();
  if (placement.is_undetermined()) {
    if (op->num_inputs() > 0) {
      placement = op->input(0)->device();
    } else {
      placement = Device(kCPU);
    }
  }

  bool ok = op->Instantiate(placement, get_suggested_stream_index(op));
  HT_RUNTIME_ERROR_IF(!ok) << "Failed to place op " << op->name()
                           << " on device " << placement;
  
  NDArrayList input_arrays;
  input_arrays.reserve(op->num_inputs());
  for (auto& input : op->inputs())
    input_arrays.push_back(_preserved_data[input->id()]);
  auto output_arrays = op->Compute(input_arrays, _runtime_ctxs);
  for (size_t i = 0; i < op->num_outputs(); i++)
    _preserved_data[op->output(i)->id()] = output_arrays[i];

  return _op_indexing[op->id()];
}

void EagerGraph::RemoveTensor(const Tensor& tensor) {
  auto& producer = _op_indexing[tensor->producer_id()];
  size_t num_outputs = MAX(producer->num_outputs(), 1);
  if (num_outputs ==
      (++_op_to_num_destructed_outputs[tensor->producer_id()])) {
    RemoveOp(producer);
  }
}

NDArray& EagerGraph::GetOrCompute(Tensor& tensor) {
  // This function should only be called from Tensor.
  // So we do not need to check the existence.
  tensor->producer()->Sync();
  return _preserved_data[tensor->id()];
}

NDArrayList EagerGraph::Run(const TensorList& fetches,
                            const Tensor2NDArrayMap&) {
  NDArrayList ret;
  ret.reserve(fetches.size());
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());
  for (auto& fetch : fetches) {
    auto it = _preserved_data.find(fetch->id());
    HT_VALUE_ERROR_IF(it == _preserved_data.end())
      << "Tensor " << fetch->name() << " cannot be found in graph " << id();
    ret.push_back(it->second);
    to_sync_op_ids.insert(fetch->producer_id());
  }
  for (auto op_id : to_sync_op_ids)
    _op_indexing[op_id]->Sync();
  return ret;
}

} // namespace graph
} // namespace hetu
