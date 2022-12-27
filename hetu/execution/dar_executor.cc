#include "hetu/execution/dar_executor.h"
#include "hetu/execution/device_placer.h"
#include "hetu/execution/run_metadata.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/dataloader.h"
#include "hetu/impl/communication/comm_group.h"
#include <queue>

namespace hetu {
namespace execution {

using hetu::operator<<;

DARExecutor::DARExecutor(const Device& local_device,
                         const DeviceGroup& device_group,
                         const TensorList& losses) {
  _exec_ctx = std::make_shared<DARExecutionContext>();
  _exec_ctx->_local_device = local_device;
  _exec_ctx->_device_group = device_group;
  // TODO: support garbage collection in OperatorPool
  bool parallel = PlaceDevices(TopoSort(OperatorPool::GetAllOps(), true));
  _exec_ctx->_global_topo_order = TopoSort(OperatorPool::GetAllOps(), true);
  HT_LOG_DEBUG << "Topo order: " << _exec_ctx->_global_topo_order;
  bool training = _exec_ctx->_global_topo_order.end() !=
    std::find_if(_exec_ctx->_global_topo_order.begin(),
                 _exec_ctx->_global_topo_order.end(),
                 [](const Operator& op) { return is_optimizer_update_op(op); });
  if (parallel) {
    _exec_ctx->_local_topo_order =
      get_local_nodes(_exec_ctx->_global_topo_order, _exec_ctx->_local_device);
    if (_exec_ctx->is_pipeline_parallel()) {
      HT_LOG_DEBUG << "Topo order of pipeline stage: "
                   << _exec_ctx->_local_topo_order;
      HT_ASSERT(!training || !losses.empty())
        << "Must provide loss when training in pipeline parallel";
    }
  } else {
    _exec_ctx->_local_topo_order = _exec_ctx->_global_topo_order;
  }
  _exec_ctx->_losses = losses;
  _exec_ctx->_training = training;
}

NDArrayList DARExecutor::Run(const TensorList& fetches,
                             const FeedDict& feed_dict,
                             const int num_micro_batches) {
  auto sub_exec = GetOrCreateSubExecutor(fetches);
  if (num_micro_batches > 1) {
    HT_ASSERT(_exec_ctx->is_pipeline_parallel())
      << "num_micro_batches > 1 only allowed in the pipeline parallelism!";
  }
  return sub_exec->Run(fetches, feed_dict, num_micro_batches);
}

bool DARExecutor::PlaceDevices(const OpList& topo_order) {
  HT_LOG_TRACE << "Device placement for topo: " << topo_order;
  // If any device_group passed to executor or any operator
  // contains 2 or more devices, then we should run in parallel.
  bool parallel = _exec_ctx->_device_group.num_devices() > 1;
  if (!parallel) {
    for (const auto& op : topo_order) {
      if (op->device_group().num_devices() > 1) {
        parallel = true;
        break;
      }
    }
  }
  // TODO: return the inserted ops during mapping and placement
  // so that we need not to call the TopoSort again
  if (parallel) {
    MapOpsToParallelDevices(topo_order, _exec_ctx->_device_group);
    OpList updated_topo_order =
      TopoSort(ExtendSubgraphWithCommunicationNodes(topo_order));
    OpList local_topo_order =
      get_local_nodes(updated_topo_order, _exec_ctx->_local_device);
    PlaceToLocalDevice(local_topo_order, _exec_ctx->_local_device);
  } else {
    PlaceToLocalDevice(topo_order, _exec_ctx->_local_device);
  }
  // TODO: return the specific parallel strategy
  return parallel;
}

std::shared_ptr<DARSubExecutor>
DARExecutor::GetOrCreateSubExecutor(const TensorList& fetches) {
  TIK(get_or_create);
  std::shared_ptr<DARSubExecutor> sub_exec = nullptr;
  // Lookup sub_executors by fetch_ids.
  TensorIdList fetch_ids(fetches.size());
  std::transform(fetches.begin(), fetches.end(), fetch_ids.begin(),
                 [](const Tensor& tensor) { return tensor->id(); });
  auto fetches_it = _fetches_to_sub_executors.find(fetch_ids);
  if (fetches_it == _fetches_to_sub_executors.end()) {
    // Lookup sub_executors by topo_ids. Here we sort fetches to
    // ensure that the topo order is deterministic.
    TensorList sorted_fetches(fetches.size());
    std::partial_sort_copy(
      fetches.begin(), fetches.end(), sorted_fetches.begin(),
      sorted_fetches.end(),
      [](const Tensor& a, const Tensor& b) { return a->id() < b->id(); });
    auto topo_order = TopoSort(sorted_fetches, true);
    OpList local_fw_topo, local_bw_topo;
    if (_exec_ctx->is_pipeline_parallel()) {
      auto parts = disentangle_forward_and_backward_nodes(
        topo_order, _exec_ctx->_losses, true);
      local_fw_topo =
        get_local_nodes(std::get<0>(parts), _exec_ctx->_local_device);
      local_bw_topo =
        get_local_nodes(std::get<1>(parts), _exec_ctx->_local_device);
      topo_order.clear();
      topo_order.reserve(local_fw_topo.size() + local_bw_topo.size());
      topo_order.insert(topo_order.end(), local_fw_topo.begin(),
                        local_fw_topo.end());
      topo_order.insert(topo_order.end(), local_bw_topo.begin(),
                        local_bw_topo.end());
    }
    // get or create a sub_executor according to the topo order
    OpIdList topo_order_ids(topo_order.size());
    std::transform(topo_order.begin(), topo_order.end(), topo_order_ids.begin(),
                   [](const Operator& op) { return op->id(); });
    auto topo_it = _topo_to_sub_executors.find(topo_order_ids);
    if (topo_it == _topo_to_sub_executors.end()) {
      // create a sub_executor
      if (_exec_ctx->is_pipeline_parallel()) {
        sub_exec = std::make_shared<PipelineDARSubExecutor>(
          _exec_ctx, topo_order, local_fw_topo, local_bw_topo);
      } else {
        sub_exec =
          std::make_shared<DefaultDARSubExecutor>(_exec_ctx, topo_order);
      }
      _topo_to_sub_executors.insert({topo_order_ids, sub_exec});
      TOK(get_or_create);
      HT_LOG_DEBUG << "Create SubExecutor for fetches " << fetches << " cost "
                   << COST_MICROSEC(get_or_create) / 1000.0 << " ms";
    } else {
      // reuse the sub_executor
      sub_exec = topo_it->second;
    }
    _fetches_to_sub_executors.insert({fetch_ids, sub_exec});
  } else {
    sub_exec = fetches_it->second;
  }

  return sub_exec;
}

DARSubExecutor::DARSubExecutor(std::shared_ptr<DARExecutionContext> exec_ctx,
                               const OpList& topo_order)
: _exec_ctx(exec_ctx), _topo_order(topo_order) {
  std::unordered_set<OpId> ops;
  ops.reserve(_topo_order.size());
  size_t num_out_edges = 0;
  for (auto& op : _topo_order) {
    ops.insert(op->id());
    num_out_edges += op->num_outputs();
    if (is_variable_op(op)) {
      _variable_ops.push_back(op);
    } else if (is_placeholder_op(op)) {
      _placeholder_ops.push_back(op);
    } else if (is_data_loader_op(op)) {
      _data_loader_ops.push_back(op);
    } else {
      _computing_ops.push_back(op);
    }
  }

  _edge_out_degrees.reserve(num_out_edges);
  for (auto& op : _topo_order) {
    for (auto& output : op->outputs())
      _edge_out_degrees[output->id()] = 0;
    for (auto& input : op->inputs())
      _edge_out_degrees[input->id()]++;
  }
}

NDArrayList DefaultDARSubExecutor::Run(const TensorList& fetches,
                                       const FeedDict& feed_dict,
                                       const int num_micro_batches) {
  Tensor2NDArrayMap edge2arr;
  std::unordered_map<TensorId, int> edge2degrees = _edge_out_degrees;
  std::unordered_set<OpId> fetch_ids;
  fetch_ids.reserve(fetches.size());
  for (const auto& fetch : fetches)
    fetch_ids.insert(fetch->id());

  // get feed in values
  for (const auto& kv : feed_dict) {
    // TODO: transfer H2D if needed
    // TODO: check shape & dtype
    edge2arr[kv.first] = kv.second;
    if (edge2arr[kv.first]->is_cpu()) {
      edge2arr[kv.first] = NDArray::cuda(edge2arr[kv.first]);
    }
  }
  for (auto& op : _placeholder_ops) {
    HT_ASSERT(edge2arr.find(op->output(0)->id()) != edge2arr.end())
      << "Cannot find values for op \"" << op->name() << "\" in feed_dict";
  }

  // get variable values
  for (auto& op : _variable_ops) {
    VariableOp& var = reinterpret_cast<VariableOp&>(op);
    edge2arr[op->output(0)->id()] = var->data();
  }

  // get dataloader values
  for (auto& op : _data_loader_ops) {
    DataloaderOp& var = reinterpret_cast<DataloaderOp&>(op);
    int idx = 0;
    for (auto it = var->dataloaders().begin(); it != var->dataloaders().end();
         ++it, ++idx) {
      edge2arr[op->output(idx)->id()] = it->second->get_arr();
    }
  }

  RuntimeContext runtime_ctx;

  // compute
  for (auto& op : _computing_ops) {
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT(it != edge2arr.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      input_vals.push_back(it->second);
      if ((--edge2degrees[in_edge->id()]) == 0 &&
          fetch_ids.find(in_edge->id()) == fetch_ids.end()) {
        edge2arr.erase(in_edge->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      if (edge2degrees[out_edge->id()] > 0 ||
          fetch_ids.find(out_edge->id()) != fetch_ids.end()) {
        edge2arr[out_edge->id()] = output_vals[i];
      }
    }
  }

  // get results
  NDArrayList results;
  for (const auto& fetch : fetches) {
    auto& op = fetch->producer();
    if (_exec_ctx->is_pipeline_parallel() &&
        op->placement() != _exec_ctx->local_device()) {
      results.push_back(NDArray());
      continue;
    }
    if (is_variable_op(op) || is_placeholder_op(op) || is_data_loader_op(op)) {
      results.push_back(edge2arr[fetch->id()]);
    } else {
      op->Sync();
      auto it = edge2arr.find(fetch->id());
      if (it != edge2arr.end())
        results.push_back(it->second);
      else
        results.push_back(NDArray());
    }
  }
  return results;
}

PipelineDARSubExecutor::PipelineDARSubExecutor(
  std::shared_ptr<DARExecutionContext> exec_ctx, const OpList& topo_order,
  const OpList& fw_topo_order, const OpList& bw_topo_order)
: DARSubExecutor(exec_ctx, topo_order),
  _fw_topo_order(fw_topo_order),
  _bw_topo_order(bw_topo_order) {
  // fw/bw compute ops
  for (auto& op : _fw_topo_order) {
    if (!is_variable_op(op) && !is_placeholder_op(op) &&
        !is_data_loader_op(op)) {
      _fw_computing_ops.push_back(op);
    }
  }
  for (auto& op : _bw_topo_order) {
    if (!is_variable_op(op) && !is_placeholder_op(op) &&
        !is_data_loader_op(op)) {
      _bw_computing_ops.push_back(op);
    }
  }
  // gradient ops
  OpList allreduce_ops;
  OpList update_ops;
  for (auto& op : _bw_computing_ops) {
    if (is_all_reduce_op(op)) {
      allreduce_ops.push_back(op);
    }
    if (is_optimizer_update_op(op)) {
      update_ops.push_back(op);
    }
  }
  // gradient consumer ops
  if (allreduce_ops.size() > 0) {
    _gradient_ops = allreduce_ops;
  } else {
    _gradient_ops = update_ops;
  }
  for (auto& gradient_op : _gradient_ops) {
    std::queue<Tensor> q;
    if (is_optimizer_update_op(gradient_op)) {
      q.push(gradient_op->out_dep_linker());
    } else {
      q.push(gradient_op->output(0));
    }
    while (!q.empty()) {
      auto& output = q.front();
      q.pop();
      for (size_t i = 0; i < output->num_consumers(); i++) {
        auto& consumer_op = output->consumer(i);
        _gradient_consumer_ops.push_back(consumer_op);
        for (size_t j = 0; j < consumer_op->num_outputs(); j++) {
          q.push(consumer_op->output(j));
        }
        if (is_optimizer_update_op(consumer_op)) {
          q.push(consumer_op->out_dep_linker());
        }
      }
    }
  }
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<int, std::vector<std::pair<bool, int>>>
PipelineDARSubExecutor::generate_gpipe_schedule(int num_stages,
                                                int num_micro_batches) {
  std::unordered_map<int, std::vector<std::pair<bool, int>>> schedule;
  // inference time: for only forward
  if (_bw_computing_ops.size() == 0) {
    for (int stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, int>> tasks;
      tasks.reserve(num_micro_batches);
      for (int step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (int stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, int>> tasks;
    tasks.reserve(2 * num_micro_batches);
    for (int step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({true, step_id});
    }
    for (int step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({false, step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<int, std::vector<std::pair<bool, int>>>
PipelineDARSubExecutor::generate_pipedream_flush_schedule(
  int num_stages, int num_micro_batches) {
  HT_ASSERT(num_micro_batches >= num_stages)
    << "num_micro_batches must bigger than num_stages in pipedream-flush!";
  std::unordered_map<int, std::vector<std::pair<bool, int>>> schedule;
  // inference time: for only forward
  if (_bw_computing_ops.size() == 0) {
    for (int stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, int>> tasks;
      tasks.reserve(num_micro_batches);
      for (int step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (int stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, int>> tasks;
    tasks.reserve(2 * num_micro_batches);
    int num_warmup_microbatches = num_stages - stage_id - 1;
    int num_microbatches_remaining =
      num_micro_batches - num_warmup_microbatches;
    // 1. warmup
    for (int step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({true, step_id});
    }
    // 2. 1F1B
    for (int step_id = 0; step_id < num_microbatches_remaining; step_id++) {
      tasks.push_back({true, num_warmup_microbatches + step_id});
      tasks.push_back({false, step_id});
    }
    // 3. cooldown
    for (int step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({false, num_microbatches_remaining + step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

void PipelineDARSubExecutor::compute_fn(
  OpList& compute_ops, Tensor2NDArrayMap& edge2arr,
  std::unordered_map<TensorId, int>& edge2degrees,
  std::unordered_map<TensorId, NDArray>& grad_accumulation,
  bool grad_accumulation_finished, std::unordered_set<OpId>& fetch_ids,
  RuntimeContext& runtime_ctx) {
  for (auto& op : compute_ops) {
    // grad accmulation for udpate_op and all_reduce_op just before update_op
    TensorId grad_id = -1;
    if (is_all_reduce_op(op) &&
        is_optimizer_update_op(op->output(0)->consumer(0))) {
      grad_id = op->input(0)->id();
    }
    if (is_optimizer_update_op(op) &&
        !is_all_reduce_op(op->input(1)->producer())) {
      grad_id = op->input(1)->id();
    }
    if (grad_id != -1) {
      if (grad_accumulation.find(grad_id) == grad_accumulation.end()) {
        grad_accumulation[grad_id] = edge2arr[grad_id];
      } else {
        grad_accumulation[grad_id] =
          NDArray::add(grad_accumulation[grad_id], edge2arr[grad_id]);
      }
      if (grad_accumulation_finished) {
        edge2arr[grad_id] = grad_accumulation[grad_id];
      } else {
        continue;
      }
    } else if (!grad_accumulation_finished) {
      bool is_consumer_op = _gradient_consumer_ops.end() !=
        std::find_if(_gradient_consumer_ops.begin(),
                     _gradient_consumer_ops.end(), [&](Operator& consumer_op) {
                       return consumer_op->id() == op->id();
                     });
      if (is_consumer_op) {
        continue;
      }
    }

    // compute
    if (!op.is_defined())
      continue; // should not happen
    HT_LOG_TRACE << "Executing op \"" << op->name() << "\"...";
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& in_edge : op->inputs()) {
      auto it = edge2arr.find(in_edge->id());
      HT_ASSERT(it != edge2arr.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << in_edge;
      input_vals.push_back(it->second);
      if ((--edge2degrees[in_edge->id()]) == 0 &&
          fetch_ids.find(in_edge->id()) == fetch_ids.end()) {
        edge2arr.erase(in_edge->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& out_edge = op->output(i);
      if (edge2degrees[out_edge->id()] > 0 ||
          fetch_ids.find(out_edge->id()) != fetch_ids.end()) {
        edge2arr[out_edge->id()] = output_vals[i];
      }
    }
  }
}

NDArrayList PipelineDARSubExecutor::Run(const TensorList& fetches,
                                        const FeedDict& feed_dict,
                                        const int num_micro_batches) {
  std::vector<Tensor2NDArrayMap> edge2arr_list(
    num_micro_batches); // edge2arr for m micro batches
  std::vector<std::unordered_map<TensorId, int>> edge2degrees_list(
    num_micro_batches,
    _edge_out_degrees); // // edge2degrees for m micro batches
  std::vector<RuntimeContext> runtime_ctx_list;
  runtime_ctx_list.reserve(
    num_micro_batches); // // runtimectx for m micro batches
  std::unordered_map<TensorId, NDArray> grad_accumulation;
  grad_accumulation.reserve(
    _variable_ops.size()); // for weights grad accumulation
  std::unordered_set<OpId> fetch_ids;
  fetch_ids.reserve(fetches.size());

  for (int i = 0; i < num_micro_batches; i++)
    runtime_ctx_list.push_back(RuntimeContext());

  for (const auto& fetch : fetches)
    fetch_ids.insert(fetch->id());

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 1. init[start]";
  // 1. init
  // get feed in values
  for (const auto& kv : feed_dict) {
    // TODO: transfer H2D if needed
    // TODO: check shape & dtype
    auto micro_batches = NDArray::split(kv.second, num_micro_batches);
    for (int i = 0; i < num_micro_batches; i++) {
      edge2arr_list[i][kv.first] = micro_batches[i];
    }
  }
  for (auto& op : _placeholder_ops) {
    for (auto& edge2arr : edge2arr_list) {
      HT_ASSERT(edge2arr.find(op->output(0)->id()) != edge2arr.end())
        << "Cannot find values for op \"" << op->name() << "\" in feed_dict";
    }
  }

  // get variable values
  for (auto& op : _variable_ops) {
    VariableOp& var = reinterpret_cast<VariableOp&>(op);
    for (auto& edge2arr : edge2arr_list) {
      edge2arr[op->output(0)->id()] = var->data();
    }
  }

  // get dataloader values
  for (auto& op : _data_loader_ops) {
    DataloaderOp& var = reinterpret_cast<DataloaderOp&>(op);
    int idx = 0;
    for (auto it = var->dataloaders().begin(); it != var->dataloaders().end();
         ++it, ++idx) {
      for (auto& edge2arr : edge2arr_list) {
        edge2arr[op->output(idx)->id()] = it->second->get_arr();
      }
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 1. init[end]";

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 2. compute[start]";
  // 2. compute
  // TODO: pp stage should support different dp degrees
  int dp = _exec_ctx->global_topo_order()[0]
             ->placement_group()
             .num_devices(); // data parallel degree
  int num_stages =
    _exec_ctx->device_group().num_devices() / dp; // get stage num
  auto schedule = generate_pipedream_flush_schedule(
    num_stages,
    num_micro_batches); // get task schedule table for pipedream-flush
  // auto schedule = generate_gpipe_schedule(num_stages, num_micro_batches); //
  // // get task schedule table for gpipe
  auto& tasks = schedule[_exec_ctx->local_device().index() /
                         dp]; // get tasks for current device
  for (std::size_t i = 0; i < tasks.size(); i++) {
    auto& task = tasks[i];
    bool is_forward = task.first;
    int micro_batch_id = task.second;
    auto& edge2arr = edge2arr_list[micro_batch_id];
    auto& edge2degrees = edge2degrees_list[micro_batch_id];
    auto& runtime_ctx = runtime_ctx_list[micro_batch_id];
    if (is_forward) {
      compute_fn(_fw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 false, fetch_ids, runtime_ctx);
    } else if (i < tasks.size() - 1) {
      compute_fn(_bw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 false, fetch_ids, runtime_ctx);
    } else {
      compute_fn(_bw_computing_ops, edge2arr, edge2degrees, grad_accumulation,
                 true, fetch_ids, runtime_ctx);
    }
    if (is_forward) {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": [micro batch "
                   << micro_batch_id << ": forward]";
    } else {
      HT_LOG_DEBUG << _exec_ctx->local_device() << ": [micro batch "
                   << micro_batch_id << ": backward]";
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 2. compute[end]";

  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 3. get results[start]";
  // 3. get results
  NDArrayList results;
  for (const auto& fetch : fetches) {
    auto& op = fetch->producer();
    if (_exec_ctx->is_pipeline_parallel() &&
        op->placement() != _exec_ctx->local_device()) {
      results.push_back(NDArray());
      continue;
    }
    if (is_variable_op(op)) {
      results.push_back(edge2arr_list[num_micro_batches - 1][fetch->id()]);
      HT_LOG_DEBUG
        << _exec_ctx->local_device() << ": fetch " << fetch
        << " is varibale op's output, get result in last micro batch updated, shape: "
        << results.back()->shape();
    } else if (is_placeholder_op(op) || is_data_loader_op(op)) {
      NDArrayList result;
      result.reserve(num_micro_batches);
      for (auto& edge2arr : edge2arr_list) {
        result.push_back(edge2arr[fetch->id()]);
      }
      results.push_back(NDArray::cat(result));
      HT_LOG_DEBUG
        << _exec_ctx->local_device() << ": fetch " << fetch
        << " is placeholder/data_loader op's output, cat result in all micro batches, shape: "
        << results.back()->shape();
    } else {
      op->Sync();
      // case 1
      auto it_grad = grad_accumulation.find(fetch->id());
      if (it_grad != grad_accumulation.end()) {
        results.push_back(it_grad->second);
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is allreduce/update op's input, get result directly in grad_accumulation, shape: "
          << results.back()->shape();
        continue;
      }
      // case 2
      if ((is_all_reduce_op(op) &&
           is_optimizer_update_op(op->output(0)->consumer(0))) ||
          is_optimizer_update_op(op)) {
        results.push_back(edge2arr_list[num_micro_batches - 1][fetch->id()]);
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is allreduce/update op's output, get result in last micro batch updated, shape: "
          << results.back()->shape();
        continue;
      }
      // case 3
      auto it = edge2arr_list[num_micro_batches - 1].find(fetch->id());
      if (it != edge2arr_list[num_micro_batches - 1].end()) {
        NDArrayList result;
        result.reserve(num_micro_batches);
        for (auto& edge2arr : edge2arr_list) {
          result.push_back(edge2arr[fetch->id()]);
        }
        results.push_back(NDArray::cat(result));
        HT_LOG_DEBUG
          << _exec_ctx->local_device() << ": fetch " << fetch
          << " is common compute op's output, cat result in all micro batches, shape: "
          << results.back()->shape();
      } else {
        results.push_back(NDArray());
        HT_LOG_DEBUG << _exec_ctx->local_device() << ": fetch " << fetch
                     << " is common compute op's output, but return NDArray()";
      }
    }
  }
  HT_LOG_DEBUG << _exec_ctx->local_device() << ": 3. get results[end]";
  return results;
}

} // namespace execution
} // namespace hetu
