#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/autocast/autocast.h"
#include "hetu/graph/recompute/recompute.h"
#include "hetu/graph/offload/activation_cpu_offload.h"

namespace hetu {
namespace graph {

Operator& DefineAndRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                         TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  if (!op_meta.device_group.empty() && std::find(_device_groups.begin(), _device_groups.end(), op_meta.device_group) == _device_groups.end())
    _device_groups.push_back(op_meta.device_group);
  // for optimization passes
  op_meta = op_meta.set_is_recompute(Recompute::enabled())
                   .set_is_cpu_offload(ActivationCPUOffload::enabled());
  return MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
}

void DefineAndRunGraph::ResetVariableDataInner(const Tensor& tensor,
                                               const Initializer& init) {
  // Mark an add-on initializer.
  _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_plan].tensor_to_exec_tensor_mapping;
    auto it = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it != tensor_to_exec_tensor_mapping.end()) {
      // The op has been instantiated in the current active graph. Also let the executable graph reset it.
      Graph::ResetVariableData(it->second, init);
    }
  }
}

NDArray DefineAndRunGraph::GetDetachedVariableDataInner(const Tensor& tensor) {
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_plan].tensor_to_exec_tensor_mapping;
    auto it_1 = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it_1 == tensor_to_exec_tensor_mapping.end()) {
      // The tensor is not in current active exec graph.
      // Question: store the data on different devices? For now, store all on CPU and return.
      auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      auto it_2 = _add_on_inits.find(tensor->id());
      // Note _add_on_inits has a higher priority than the original tensor initializer.
      if (it_2 != _add_on_inits.end()) {
        HT_LOG_TRACE << "The data is reset, but not in current active exec graph, "
          << "so we get the data of the variable from the DefineAndRun graph.";
        it_2->second->Init(ret);
      } else {
        HT_LOG_TRACE << "The data is not in current active exec graph, " 
          << "so we get the data of the variable from its initializer.";
        if (tensor->has_distributed_states())
          dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
        else
          dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
      }
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    } else {
      // The op has been instantiated in the current active graph. Let the executable graph handle it.
      if (!it_1->second->producer()->device_group().contains(impl::comm::GetLocalDevice())) {
        HT_LOG_TRACE << "The data is not locate at local executable graph, return an empty NDArray.";
        return NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      }
      auto ret = Graph::GetDetachedVariableData(it_1->second);
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    }  
  } else {
    auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
    auto it = _add_on_inits.find(tensor->id());
    // Note _add_on_inits has a higher priority than the original tensor initializer.
    if (it != _add_on_inits.end()) {
      HT_LOG_TRACE << "No active exec graph yet. The data is reset, " 
        << "so we get the data of the variable from the DefineAndRun graph.";
      it->second->Init(ret);
    } else {
      HT_LOG_TRACE << "No active exec graph yet, "
        << "so we get the data of the variable from its initializer.";
      if (tensor->has_distributed_states())
        dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
      else
        dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
    }
    Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
    stream.Sync();
    return ret;
  }
}

DeviceGroup DefineAndRunGraph::GetVariableDeviceGroupInner(const Tensor& tensor) {
  auto& device_group = tensor->producer()->device_group();
  HT_RUNTIME_ERROR_IF(device_group.empty()) << "You are getting an empty device group, please ensure you have set "
    << tensor->producer() << " a device group before!";
  return device_group;
}

void DefineAndRunGraph::Instantiate(const Tensor2ShapeMap& shape_plan) {

  auto exec_graph_num = _exec_graph_plan_pool.size();
  Tensor2ShapeMap exec_shape_plan;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  auto exec_graph = Graph::_make_new_graph<ExecutableGraph>(name() + "_executable_" + std::to_string(exec_graph_num));
  
  exec_shape_plan.reserve(shape_plan.size());
  op_to_exec_op_mapping.reserve(_init_capacity);
  tensor_to_exec_tensor_mapping.reserve(_init_capacity);

  Graph::push_graph_ctx(exec_graph->id());

  // assign pp stages
  exec_graph->SetStages(_device_groups);

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  auto put_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    auto plan_it = shape_plan.find(tensor->id());
    if (plan_it != shape_plan.end())
      exec_tensor->set_shape(plan_it->second);
    // Else: if the tensor to instantiate is not in the shape plan, 
    // then the tensor won't be used in the exec graph.
    // So we just leave its shape to its default.
    exec_tensor->set_is_grad(tensor->is_grad());
    exec_shape_plan[exec_tensor->id()] = exec_tensor->shape();
    tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    // assign symbolic shape
    if (tensor->symbolic())
      exec_tensor->set_symbolic_shape(tensor->symbolic_shape());
    auto it = _add_on_inits.find(tensor->id());
    if (it != _add_on_inits.end()) {
      Graph::ResetVariableData(exec_tensor, *it->second);
      /*
      // 所有的exec graph pool都需要共享，不能删除
      // _add_on_inits.erase(tensor->id());
      */
      // 2023.12.9修改：之后考虑要切换plan，仅第一次使用_add_on_init
    }
  };

  auto topo = topo_order();
  HT_LOG_DEBUG << "Instantiating a " << type() << " graph with topo " << topo;
  std::unordered_map<int, Tensor> transfer_map;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (op_to_exec_op_mapping.find(op->id()) != op_to_exec_op_mapping.end())
      continue;
    HT_LOG_DEBUG << "Creating an executable version of op " << op;

    TensorList exec_inputs, exec_in_deps;
    std::tie(exec_inputs, exec_in_deps) =
      Operator::transform_each_input_tensor(op, get_exec_input);

    auto autocast_id = AutoCast::cur_autocast_ctx();
    if (autocast_id != UINT64_MAX) {
      auto autocast = AutoCast::GetAutoCast(autocast_id);
      if (autocast.enabled()) {
        DataType datatype = DataType::UNDETERMINED;
        if (autocast.cast_type() != DataType::UNDETERMINED)
          datatype = autocast.cast_type();

        if (datatype != DataType::UNDETERMINED) {
          auto optype = op->type();
          if (is_optimizer_update_op(op) || is_host_to_device_op(op) || is_device_to_host_op(op) || is_data_transfer_op(op)) {}
          else {
            for (int i = 0; i < exec_inputs.size(); ++i) {
              if ((is_variable_op(exec_inputs[i]->producer()) || is_placeholder_op(exec_inputs[i]->producer())) &&
                  exec_inputs[i]->dtype() != datatype && 
                  (exec_inputs[i]->dtype() == DataType::BFLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT32 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT64)) {
                if (transfer_map.find(exec_inputs[i]->id()) != transfer_map.end()) {
                  HT_LOG_DEBUG << "Map" << &transfer_map << "ReUse:" << exec_inputs[i]->id() << "->" << transfer_map[exec_inputs[i]->id()]->id();
                  exec_inputs[i] = transfer_map[exec_inputs[i]->id()];
                }
                else {
                  auto& exec_op = Graph::MakeOp(std::make_shared<DataTransferOpImpl>(datatype, exec_inputs[i]->device()),
                                  {exec_inputs[i]}, OpMeta().set(op->op_meta()).set_name(op->name() + "_transfer"), *exec_graph);
                  HT_LOG_DEBUG << "Map" << &transfer_map << "Insert:" << exec_inputs[i]->id() << "->" << exec_op->output(0)->id();
                  transfer_map[exec_inputs[i]->id()] = exec_op->output(0);
                  exec_inputs[i] = exec_op->output(0);
                }
                exec_graph->RecordTensorShape(exec_inputs[i]->id(), exec_inputs[i]->shape());
              }
            }
          }
        }
      }
    }

    auto& exec_op = Graph::MakeOp(
      op->_body, std::move(exec_inputs),
      OpMeta().set(op->op_meta()).set_extra_deps(std::move(exec_in_deps)),
      *exec_graph);
    if (_parameter_ops.find(op->id()) != _parameter_ops.end())
      Graph::MarkAsParameter(exec_op);

    Operator::for_each_output_tensor_pair(op, exec_op, put_exec_output);
    if (is_placeholder_op(op) || is_variable_op(op)) {
      if (op->output(0)->has_distributed_states())
        exec_op->output(0)->set_distributed_states(op->output(0)->get_distributed_states());
    }
    op_to_exec_op_mapping[op->id()] = exec_op;
  }

  // assign fw_op_id map
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    auto& exec_op = op_to_exec_op_mapping[op->id()];
    if (op->fw_op_id() != -1) {
      exec_op->set_fw_op_id(op_to_exec_op_mapping[op->fw_op_id()]->id());
    } 
  }

  // assign initial shape plan
  exec_graph->InitShapePlan(std::move(exec_shape_plan));
  
  // wrap up all of this as an exec graph plan
  _exec_graph_plan_pool.emplace_back(std::move(exec_graph), 
                                     std::move(op_to_exec_op_mapping),
                                     std::move(tensor_to_exec_tensor_mapping));

  Graph::pop_graph_ctx();
}

// TODO: merge two `Run` func
NDArrayList DefineAndRunGraph::Run(const TensorList& fetches,
                                   const FeedDict& feed_dict) {
  HT_RUNTIME_ERROR << "NotImplementedError";
  /*
  bool has_uninstantiated_ops =
    std::any_of(fetches.begin(), fetches.end(), [&](const Tensor& fetch) {
      return _op_to_exec_op_mapping.find(fetch->producer_id()) ==
        _op_to_exec_op_mapping.end();
    });
  if (has_uninstantiated_ops)
    Instantiate();
  TensorList exec_fetches;
  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    exec_fetches.push_back(_tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  FeedDict exec_feed_dict;
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict)
    exec_feed_dict[_tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  return _exec_graph->Run(exec_fetches, exec_feed_dict);
  */
}

NDArrayList DefineAndRunGraph::Run(const Tensor& loss, const TensorList& fetches,
                                   const FeedDict& feed_dict, const int num_micro_batches) {

  auto local_device = hetu::impl::comm::GetLocalDevice(); // only for debug use
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph begin...";

  // get feed dict shape
  Tensor2ShapeMap feed_dict_shape;
  for (const auto& kv : feed_dict) {
    if (!kv.second.is_defined()) 
      continue; 
    // TODO: use NDArrayMeta::split instead, but currently no support for arg chunk_num
    auto micro_batches = NDArray::split(kv.second, num_micro_batches);
    // currently assume that all micro batches have the same shape
    feed_dict_shape[kv.first] = micro_batches[0]->shape();
  }

  // 匹配shape plan
  size_t next_active_plan;
  size_t plan_scanned = 0;
  // 逆向扫描能够更快地匹配到
  for (auto plan_it = _shape_plan_pool.rbegin(); plan_it != _shape_plan_pool.rend(); ++plan_it)  {
    const auto& shape_plan = *plan_it;
    bool shape_plan_matched = true;
    for (const auto& kv : feed_dict) {
      if (!kv.second.is_defined()) continue;
      auto it = shape_plan.find(kv.first);
      // 1、有可能是feed dict发生了改变（在依据topo生成的shape plan中没有feed dict）
      // 2、有可能是feed dict的shape发生了改变（shape对不上）
      HT_LOG_TRACE << local_device << ": shape plan is " << shape_plan << " and key to match is "
        << kv.first << ":" << feed_dict_shape[kv.first];
      if (it == shape_plan.end() || it->second != feed_dict_shape[kv.first]) {
        shape_plan_matched = false;
        break;
      }
    }
    if (shape_plan_matched) 
      break;
    plan_scanned++;
  }

  // 需要新增shape plan到shape plan pools
  // 同时也需要新增exec graph plan到exec graph plan pools
  if (plan_scanned == _shape_plan_pool.size()) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new shape plan and an exec graph to the pool begin...";
    Tensor2ShapeMap shape_plan;
    // feed_dict中的shape是确定的
    for (const auto& kv : feed_dict) {
      shape_plan[kv.first] = feed_dict_shape[kv.first];
    }
    auto is_feed_dict_op = [&](const Operator& op) -> bool {
      return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
        return feed_dict.find(tensor->id()) != feed_dict.end();
      });
    };
    OpRefList topo = Graph::TopoSort(fetches, -1, is_feed_dict_op);
    HT_LOG_DEBUG << local_device << ": global topo before deducing shape plan is " << topo;
    RuntimeContext runtime_ctx(topo.size());
    // 扫描global topo并推导新的shape plan
    for (auto& op_ref : topo) {
      auto& op = op_ref.get();
      // HT_LOG_DEBUG << local_device << ": " << op << " deducing shape...";
      // 设置placeholder（也有可能是中间的算子——具体要看feed_dict喂的是什么算子）的symbolic shape
      bool handle_feed_dict_op = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
        auto it = feed_dict.find(tensor->id());
        if (it != feed_dict.end()) {
          if (tensor->symbolic() && is_SyShape_leaf(tensor->symbolic_shape())) {
            tensor->set_symbolic_shape(feed_dict_shape[tensor->id()]);
            HT_LOG_DEBUG << local_device << ": set symbolic shape of " << op 
              << " feed_dict tensor to " << feed_dict_shape[tensor->id()];
          }
          // 不需要给DefineAndRun graph中的tensor设置shape
          // 有效的shape信息都在shape plan中
          // HT_LOG_DEBUG << local_device << ": " << tensor << " set shape " << feed_dict_shape[tensor->id()];
          // tensor->set_shape(feed_dict_shape[tensor->id()]);
          return true;
        }
        return false;
      });
      if (handle_feed_dict_op)
        continue;
      HTShapeList input_shapes;
      input_shapes.reserve(op->num_inputs());
      for (const auto& input : op->inputs()) {
        auto it = shape_plan.find(input->id());
        HT_ASSERT(it != shape_plan.end()) 
          << "Something wrong, can't find the input shape from the current shape plan!";
        input_shapes.push_back(it->second);
      }
      HTShapeList output_shapes = op->InferShape(input_shapes, runtime_ctx);
      auto output_shapes_size = output_shapes.size();
      for (size_t i = 0; i < output_shapes_size; i++) {
        // 设置symbolic shape叶子节点的shape
        // 其相关联的非叶子的symbolic shape可以直接由计算链条获得新的shape
        if (op->output(i)->symbolic()) {
          HT_LOG_DEBUG << local_device << ": op " << op 
            << " output " << i << " has " << op->output(i)->symbolic_shape();
          if (is_SyShape_leaf(op->output(i)->symbolic_shape())) {
            op->output(i)->set_symbolic_shape(output_shapes[i]);
            HT_LOG_DEBUG << local_device << ": set symbolic shape of " << op 
              << " output " << i << " to " << output_shapes[i];
          }
        }
        // 不需要给DefineAndRun graph中的tensor设置shape
        // 有效的shape信息都在shape plan中
        // HT_LOG_DEBUG << local_device << ": " << op->output(i) << " shape " << output_shapes[i];
        // op->output(i)->set_shape(output_shapes[i]);
        auto it = shape_plan.find(op->output(i)->id());
        HT_ASSERT(it == shape_plan.end()) 
          << "Something wrong, the output shape should't exist in the current shape plan";
        shape_plan.insert(std::make_pair(op->output(i)->id(), std::move(output_shapes[i]))); // move constructor
      }
    }
    Instantiate(shape_plan);
    // TODO
    // 1、根据topo和shape plan，指定新的并行方案
    _shape_plan_pool.push_back(std::move(shape_plan));
    next_active_plan = plan_scanned;
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new shape plan and an exec graph to the pool end...";
  } else {
    // 因为是从后到前扫描
    next_active_plan = _shape_plan_pool.size() - 1 - plan_scanned;
  }

  // 需要切换plan
  if (!_is_active || _active_plan != next_active_plan) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan begin...";
    // TODO
    // 1、切换到新的并行方案
    // 2、将原先_active_plan的ckpt重新分配到新方案的设备上，这样才能继续训练
    _is_active = true;
    _active_plan = next_active_plan;
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan end...";
  }
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph end...";

  // TODO: 目前暂未考虑fetches也发生变化，先只关注shape
  /*
  bool has_uninstantiated_ops =
    std::any_of(fetches.begin(), fetches.end(), [&](const Tensor& fetch) {
      return _op_to_exec_op_mapping.find(fetch->producer_id()) ==
        _op_to_exec_op_mapping.end();
    });
  if (has_uninstantiated_ops)
    Instantiate();
  */

  // 运行挑选出的active exec graph
  auto& exec_graph = _exec_graph_plan_pool[_active_plan].exec_graph;
  auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_plan].op_to_exec_op_mapping;
  auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_plan].tensor_to_exec_tensor_mapping;
  auto& exec_loss = tensor_to_exec_tensor_mapping[loss->id()]; 
  TensorList exec_fetches;
  FeedDict exec_feed_dict;

  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    exec_fetches.push_back(tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict)
    exec_feed_dict[tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  HT_LOG_TRACE << "the active exec graph start running..." ;
  return exec_graph->Run(exec_loss, exec_fetches, exec_feed_dict, num_micro_batches);
}

} // namespace graph
} // namespace hetu
