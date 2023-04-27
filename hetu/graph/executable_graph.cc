#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/data_transfer.h"

namespace hetu {
namespace graph {

Operator& ExecutableGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                       TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  return MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
}

bool ExecutableGraph::MapOpsToParallelDevices(
  const DeviceGroup& placement_group) {
  HT_NOT_IMPLEMENTED;
  return true;
}

bool ExecutableGraph::Instantiate(const TensorList& fetches,
                                  const Device& preferred_device) {
  auto is_op_instantiated = [&](const Operator& op) -> bool {
    return !op->placement().is_undetermined();
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);
  HT_LOG_TRACE << "Instantiating ops: " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (!op->placement().is_undetermined())
      continue;

    Device placement =
      is_device_to_host_op(op) ? Device(kCPU) : preferred_device;
    StreamIndex stream_id = get_suggested_stream_index(op);
    HT_LOG_TRACE << "Instantiating op " << op << " (placement=" << placement
                 << ", stream_index=" << stream_id << ")";
    bool ok = op->Instantiate(placement, stream_id);
    if (!ok && !placement.is_cpu()) {
      HT_LOG_WARN << "Failed to instantiate op " << op << " on " << placement
                  << ". Will try to instantiate it on the host device.";
      placement = Device(kCPU);
      ok = op->Instantiate(placement, stream_id);
    }
    HT_VALUE_ERROR_IF(!ok) << "Failed to instantiate op " << op << " on "
                           << placement;

    // add transfer ops
    for (size_t i = 0; i < op->num_inputs(); i++) {
      auto& input = op->input(i);
      if (input->placement() != placement) {
        HT_RUNTIME_ERROR_IF(!input->placement().local())
          << "Please use P2P communication to fetch remote input";

        auto& input_op = input->producer();

        Tensor transferred_input;
        StreamIndex transfer_stream_id;
        if (input->placement().is_cpu()) {
          transferred_input = MakeDataH2DOp(placement, input);
          transfer_stream_id = kH2DStream;
        } else if (placement.is_cpu()) {
          transferred_input = MakeDataD2HOp(placement, input);
          transfer_stream_id = kD2HStream;
        } else {
          // TODO: support cuda memcpy across processes
          HT_NOT_IMPLEMENTED << "We should use NCCL for P2P communication.";
          __builtin_unreachable();
        }
        auto& transfer_op = transferred_input->producer();
        if (!input_op->placement_group().empty())
          transfer_op->MapToParallelDevices(input_op->placement_group());
        transfer_op->Instantiate(placement, transfer_stream_id);
        ReplaceInput(op, i, transferred_input);
      }
    }
  }

  return true;
}

NDArrayList ExecutableGraph::Run(const TensorList& fetches,
                                 const FeedDict& feed_dict) {
  // TODO: For each pair of `fetches` and `feed_dict`,
  // deduce the optimal execution plan, and cache it.
  for (auto& fetch : fetches) {
    Instantiate(fetches, kCUDA);
    // if (fetch->placement().is_undetermined()) {
    //   Instantiate(fetches, kCUDA);
    //   break;
    // }
  }

  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_computed);

  RuntimeContext runtime_ctx(topo.size());
  Tensor2NDArrayMap tensor2data;
  tensor2data.reserve(topo.size());
  tensor2data.insert(feed_dict.begin(), feed_dict.end());
  NDArrayList results(fetches.size());
  std::unordered_map<TensorId, size_t> fetch_indices;
  for (size_t i = 0; i < fetches.size(); i++)
    fetch_indices[fetches.at(i)->id()] = i;
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());

  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    // Question: Is it possible that some outputs are fed in
    // while the rest are not?
    bool computed = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
    if (computed)
      continue;

    NDArrayList inputs;
    inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++) {
      // TODO: Support async transfer. And this could be checked for once.
      auto& data = tensor2data[op->input(i)->id()];
      if (data->device() != op->input(i)->placement() ||
          data->dtype() != op->input(i)->dtype()) {
        tensor2data[op->input(i)->id()] =
          NDArray::to(data, op->input(i)->placement(), op->input(i)->dtype(),
                      kBlockingStream);
      }
      inputs.push_back(tensor2data[op->input(i)->id()]);
    }
    auto outputs = op->Compute(inputs, runtime_ctx);
    for (size_t i = 0; i < outputs.size(); i++) {
      tensor2data.insert({op->output(i)->id(), outputs[i]});
      auto it = fetch_indices.find(op->output(i)->id());
      if (it != fetch_indices.end()) {
        results[it->second] = outputs[i];
        to_sync_op_ids.insert(op->id());
      }
    }
    // TODO: remove inputs that are no longer used
  }
  for (auto op_id : to_sync_op_ids) {
    _op_indexing[op_id]->Sync();
  }
  return results;
}

} // namespace graph
} // namespace hetu
