#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/autocast/autocast.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"

namespace hetu {
namespace graph {

// changing parallel plan
static size_t change_parallel_test_case = 0;

Operator& DefineAndRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                         TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  // init multi device_groups for multi ds
  if (op_meta.device_groups.size() == NUM_STRATEGY) {
    if (_multi_device_groups.empty()) {
      _multi_device_groups.resize(NUM_STRATEGY, DeviceGroupList());
    }
    for (size_t i = 0; i < NUM_STRATEGY; i++) {
      if (!op_meta.device_groups[i].empty() && std::find(_multi_device_groups[i].begin(), _multi_device_groups[i].end(), op_meta.device_groups[i]) == _multi_device_groups[i].end())
        _multi_device_groups[i].push_back(op_meta.device_groups[i]);
    }
  }

  // HT_LOG_TRACE << name() << " make op: " << op_meta.name;
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

void DefineAndRunGraph::Instantiate(const OpRefList& topo,
                                    Tensor2ShapeMap& shape_plan) {

  // Test Case: 切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      InstantiateTestCase(topo, shape_plan);
      change_parallel_test_case += 1;
      return;
    }
  }

  // initializations of the exec plan
  auto exec_graph_num = _exec_graph_plan_pool.size();
  Tensor2ShapeMap exec_shape_plan;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  auto origin_param_buffer = std::make_shared<ParamBuffer>("origin_param_buffer");
  auto transfer_param_buffer = std::make_shared<ParamBuffer>("transfer_param_buffer");
  Tensor2TensorMap transfer_map;
    
  exec_shape_plan.reserve(shape_plan.size());
  op_to_exec_op_mapping.reserve(_init_capacity);
  tensor_to_exec_tensor_mapping.reserve(_init_capacity);

  // initializations of the exec graph
  auto local_device = hetu::impl::comm::GetLocalDevice();
  auto exec_graph = Graph::_make_new_graph<ExecutableGraph>(name() + "_executable_" + std::to_string(exec_graph_num));
  exec_graph->NUM_STRATEGY = NUM_STRATEGY;
  exec_graph->CUR_STRATEGY_ID = CUR_STRATEGY_ID;
  Graph::push_graph_ctx(exec_graph->id());

  // assign pp stages
  exec_graph->SetStages(_multi_device_groups[CUR_STRATEGY_ID]);

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  // todo: just use multi_ds[cur_strategy_id] which was deduced in define_and_run_graph
  // executable_graph needn't deduce states again!
  auto handle_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    HT_LOG_DEBUG << "handle mapping of tensor " << tensor->id() << " " << tensor;
    // 1)、assign tensor mapping
    tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    // 2)、assign shape
    auto plan_it = shape_plan.find(tensor->id());
    // The shape plan will be expanded step by step
    if (plan_it != shape_plan.end()) {
      exec_tensor->set_shape(plan_it->second);
    } else {
      shape_plan[tensor->id()] = exec_tensor->shape();
    }
    exec_shape_plan[exec_tensor->id()] = exec_tensor->shape();
    HT_LOG_DEBUG << "assign exec tensor " << exec_tensor << " shape " << exec_tensor->shape();
    exec_tensor->set_is_grad(tensor->is_grad());
    // 3)、assign symbolic shape
    if (tensor->symbolic()) {
      exec_tensor->copy_symbolic_shape(tensor->symbolic_shape());
      if (is_SyShape_leaf(exec_tensor->symbolic_shape())) {
        exec_tensor->set_symbolic_shape(exec_tensor->shape());
      }
    }
    // 4)、assign distributed_states
    // just copy distributed_states here
    exec_tensor->set_multi_distributed_states(tensor->multi_distributed_states());
    // 5)、assign add on inits
    auto it = _add_on_inits.find(tensor->id());
    if (_run_level != RunLevel::TOPO && it != _add_on_inits.end()) {
      Graph::ResetVariableData(exec_tensor, *it->second);
      // 考虑要切换plan，仅第一次使用_add_on_inits
      // 之后会使用热切换
      _add_on_inits.erase(tensor->id());
    }
    // 6)、assign param 
    // 目前只是记录而并不会alloc
    if (_parameter_ops.find(tensor->producer()->id()) != _parameter_ops.end()
        && exec_tensor->producer()->device_group().contains(local_device)) {
      origin_param_buffer->AddTensor(exec_tensor);
      /*
      exec_tensor->set_placement_group(exec_tensor->producer()->device_group());
      exec_tensor->set_placement(local_device);
      Graph::AllocVariableData(exec_tensor);
      */
    }
  };

  HT_LOG_DEBUG << "Instantiating a " << type() << " graph with topo " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    HT_LOG_DEBUG << "Creating an executable version of op " << op << " begin...";;

    // 前处理
    // 1、获取exec op的inputs
    // 2、进行autocast
    TensorList exec_inputs, exec_in_deps;
    std::tie(exec_inputs, exec_in_deps) = Operator::transform_each_input_tensor(op, get_exec_input);
    auto autocast_id = AutoCast::cur_autocast_ctx();
    if (autocast_id != UINT64_MAX) {
      auto autocast = AutoCast::GetAutoCast(autocast_id);
      if (autocast.enabled()) {
        DataType datatype = DataType::UNDETERMINED;
        if (autocast.cast_type() != DataType::UNDETERMINED)
          datatype = autocast.cast_type();
        if (datatype != DataType::UNDETERMINED) {
          auto optype = op->type();
          if (is_optimizer_update_op(op) || is_host_to_device_op(op) || is_device_to_host_op(op) || is_data_transfer_op(op)) {
            // seems nothing to do
          } else {
            for (int i = 0; i < exec_inputs.size(); ++i) {
              if ((is_variable_op(exec_inputs[i]->producer()) || is_placeholder_op(exec_inputs[i]->producer())) &&
                  exec_inputs[i]->dtype() != datatype && 
                  (exec_inputs[i]->dtype() == DataType::BFLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT32 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT64)) {
                if (transfer_map.find(exec_inputs[i]->id()) != transfer_map.end()) {
                  HT_LOG_DEBUG << "Map " << &transfer_map << " reuse: " << exec_inputs[i]->id() << " -> " << transfer_map[exec_inputs[i]->id()]->id();
                  exec_inputs[i] = transfer_map[exec_inputs[i]->id()];
                } else {
                  auto& exec_op = Graph::MakeOp(std::make_shared<DataTransferOpImpl>(datatype, exec_inputs[i]->device()),
                                  {exec_inputs[i]}, OpMeta().set(exec_inputs[i]->producer()->op_meta()).set_name(exec_inputs[i]->producer()->name() + "_transfer").set_is_deduce_states(false), *exec_graph);
                  HT_LOG_DEBUG << "Map " << &transfer_map << " insert: " << exec_inputs[i]->id() << " -> " << exec_op->output(0)->id();
                  exec_shape_plan[exec_op->output(0)->id()] = exec_op->output(0)->shape();
                  exec_op->output(0)->set_multi_distributed_states(op->input(i)->multi_distributed_states()); // walkaround: set here by hand
                  if (_parameter_ops.find(op->input(i)->producer()->id()) != _parameter_ops.end()
                      && exec_inputs[i]->producer()->device_group().contains(local_device)) {
                    transfer_param_buffer->AddTensor(exec_op->output(0));
                  }
                  transfer_map[exec_inputs[i]->id()] = exec_op->output(0);
                  exec_inputs[i] = exec_op->output(0);
                }
              }
            }
          }
        }
      }
    }

    // 核心部分
    // only deduce multi ds for define_and_run_graph, and copy directly for executable_graph
    auto& exec_op = Graph::MakeOp(
      op->_body, std::move(exec_inputs),
      OpMeta().set(op->op_meta()).set_is_deduce_states(false).set_extra_deps(std::move(exec_in_deps)),
      *exec_graph);

    // 后处理
    // 1、建立op和exec_op的映射
    // 2、标记parameter（同时给即将创建的exec graph预先设置ParamBuffer）
    // 3、设置tensor的shape和distributed_states
    op_to_exec_op_mapping[op->id()] = exec_op;
    if (_parameter_ops.find(op->id()) != _parameter_ops.end()) {
      Graph::MarkAsParameter(exec_op);
    }
    Operator::for_each_output_tensor_pair(op, exec_op, handle_exec_output);

    HT_LOG_DEBUG << "Creating an executable version of op " << op << " end...";
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

  // assign param buffer and transfer map
  exec_graph->_origin_param_buffer = std::move(origin_param_buffer);
  exec_graph->_transfer_param_buffer = std::move(transfer_param_buffer);
  exec_graph->_transfer_map = std::move(transfer_map);
  
  // wrap up all of this as an exec graph plan
  _exec_graph_plan_pool.emplace_back(std::move(exec_graph), 
                                     std::move(op_to_exec_op_mapping),
                                     std::move(tensor_to_exec_tensor_mapping),
                                     CUR_STRATEGY_ID);

  Graph::pop_graph_ctx();

  // Test Case: 切换并行方案
  if (env != nullptr) {
    if (std::string(env) == "PRECISION" || std::string(env) == "COST") {
      change_parallel_test_case += 1;
    }
  }
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

// 每次调用run都会从当前的define graph中
// 生成/使用之前生成过的一个exec graph
// 而只有当：
// 1、并行策略 2、fetch的tensor 3、feed_dict的shape 
// 与cache的某一个重合时，才会复用
// 目前的写法下，我们认为并行策略已经在python端选择好了然后再传进来
NDArrayList DefineAndRunGraph::Run(const Tensor& loss, const TensorList& fetches,
                                   const FeedDict& feed_dict, const int num_micro_batches,
                                   const int cur_strategy_id, RunLevel run_level) {
  _run_level = run_level;
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
  size_t plan_size = _shape_plan_pool.size();
  size_t plan_scanned = 0;
  HT_ASSERT(_exec_graph_plan_pool.size() == plan_size)
    << "something wrong, the sizes of exec_graph_plan_pool and shape_plan_pool are mismatched";
  // 逆向扫描能够更快地匹配到
  for (auto i = static_cast<int32_t>(plan_size) - 1; i >= 0; --i)  {
    const auto& shape_plan = _shape_plan_pool[i];
    const auto& exec_graph_plan = _exec_graph_plan_pool[i];
    bool plan_matched = true;
    // 先看strategy匹配不
    if (static_cast<size_t>(cur_strategy_id) != exec_graph_plan.strategy_id) {
      plan_matched = false;
    }
    // 再看fetch匹配不
    for (const auto& fetch : fetches) {
      if (std::find(exec_graph_plan.fetches.begin(), exec_graph_plan.fetches.end(), fetch) == exec_graph_plan.fetches.end()) {
        HT_LOG_TRACE << local_device << ": exec_graph_plan fetches are " << exec_graph_plan.fetches 
          << " and the mismatch fetch is " << fetch;
        plan_matched = false;
        break;
      }
    }
    // 再看feed_dict匹配不
    for (const auto& kv : feed_dict) {
      if (!kv.second.is_defined()) continue;
      auto it = shape_plan.find(kv.first);
      // 1、有可能是feed_dict发生了改变（在依据topo生成的shape plan中没有feed dict）
      // 2、有可能是feed_dict的shape发生了改变（shape对不上）
      HT_LOG_TRACE << local_device << ": shape plan is " << shape_plan << " and key to match is "
        << kv.first << ":" << feed_dict_shape[kv.first];
      if (it == shape_plan.end() || it->second != feed_dict_shape[kv.first]) {
        plan_matched = false;
        break;
      }
    }
    if (plan_matched) {
      HT_LOG_TRACE << local_device << ": plan matched";
      break;
    }
    plan_scanned++;
  }

  // todo: 确定distributed states后，再推导shape，再根据{ds + shape}来判断是否要创建新的exec_graph
  // 需要新增shape plan到shape plan pools
  // 同时也需要新增exec graph plan到exec graph plan pools
  if (plan_scanned == plan_size) {
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

    // deprecated, but maybe useful some day
    // *now move all the logic of inferring shape and distributed_states to Instantiate()
    // that is because MakeOp can handle most of the cases automatically
    /*
    RuntimeContext runtime_ctx(topo.size());
    // std::unordered_set<TensorId> params;
    // 扫描global topo并推导新的shape plan
    for (auto& op_ref : topo) {
      auto& op = op_ref.get();
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
      if (handle_feed_dict_op) {
        continue;
      }
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
        HT_LOG_WARN << local_device << ": " << op->output(i) << " shape " << output_shapes[i];
        // op->output(i)->set_shape(output_shapes[i]);
        auto it = shape_plan.find(op->output(i)->id());
        HT_ASSERT(it == shape_plan.end()) 
          << "Something wrong, the output shape should't exist in the current shape plan";
        shape_plan.insert(std::make_pair(op->output(i)->id(), std::move(output_shapes[i]))); // move constructor
      }
    }
    */

    // set define_and_run_graph.CUR_STRATEGY_ID = dst id, 
    // and then do instantiate, if not set, use default id = 0
    CUR_STRATEGY_ID = static_cast<size_t>(cur_strategy_id);
    HT_LOG_DEBUG << "use CUR_STRATEGY_ID = " << CUR_STRATEGY_ID << " for new executable graph...";
    // Instantiate会将新的exec_graph_plan加入pool中
    Instantiate(topo, shape_plan);
    _shape_plan_pool.push_back(std::move(shape_plan));
    // 补上fetches
    auto& new_plan = _exec_graph_plan_pool.back();
    new_plan.fetches = fetches;
    // 新的plan就是pool里的最后一个
    next_active_plan = plan_scanned; // _exec_graph_plan_pool.size() - 1
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new shape plan and an exec graph to the pool end...";
  } else {
    // 因为是从后到前扫描
    next_active_plan = _shape_plan_pool.size() - 1 - plan_scanned;
  }

  // 需要切换plan
  if (!_is_active || _active_plan != next_active_plan) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan begin...";
    // 热切换
    if (_is_active) {
      auto key = std::make_pair(_active_plan, next_active_plan);
      if (_switcher_pool.find(key) == _switcher_pool.end()) {
        _switcher_pool[key] = std::make_shared<SwitchExecGraph>(this, _active_plan, next_active_plan);
      }
      // 默认的切换状态设置
      auto switch_mode = SWITCH_MODE::SWITCH_TRANSFER_PARAM;
      auto switch_level = SWITCH_LEVEL::EXEC;
      // 如果旧的exec graph没开AMP
      // 那么只能切换origin param buffer
      if (_exec_graph_plan_pool[_active_plan].exec_graph->_transfer_param_buffer->IsEmpty()) {
        switch_mode = SWITCH_MODE::SWITCH_ORIGIN_PARAM;
      }
      // 如果旧的exec graph只是建立topo
      // 那么其必然没有alloc
      // 此时无法做实际的热切换
      // 而是建立热切换的topo
      if (_exec_graph_plan_pool[_active_plan].exec_graph->_run_level == RunLevel::TOPO) {
        switch_level = SWITCH_LEVEL::TOPO;
      }
      // 如果新的exec graph只是建立topo
      // 但旧的exec graph已经alloc/grad/update了
      // 我们目前禁止这样做（以防出现一些bug）
      else {
        if (_run_level == RunLevel::TOPO) {
          HT_RUNTIME_ERROR << "graph with RunLevel::TOPO "
            << "should only followed behind graph with RunLevel::TOPO right now";
        }
      }
      _switcher_pool[key]->SwitchParams(switch_mode, switch_level);
    }
    _is_active = true;
    _active_plan = next_active_plan;
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan end...";
  }
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph end...";

  // Test Case: 切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      return {};
    }
  }

  // 运行挑选出的active exec graph
  auto& exec_graph = _exec_graph_plan_pool[_active_plan].exec_graph;
  auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_plan].op_to_exec_op_mapping;
  auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_plan].tensor_to_exec_tensor_mapping;
  auto& exec_loss = tensor_to_exec_tensor_mapping[loss->id()]; 
  TensorList exec_fetches;
  FeedDict exec_feed_dict;

  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    HT_ASSERT(tensor_to_exec_tensor_mapping.find(fetch->id()) != tensor_to_exec_tensor_mapping.end())
      << "can't find fetch tensor " << fetch << " in the mapping";
    exec_fetches.push_back(tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict) {
    if (tensor_to_exec_tensor_mapping.find(kv.first) == tensor_to_exec_tensor_mapping.end()) {
      HT_LOG_DEBUG << "feed tensor " << kv.first << " is not used in the exec graph"
        << ", so we just skipped it";
      continue;
    }
    exec_feed_dict[tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  }
  HT_LOG_INFO << exec_graph->name() << " start running..." ;
  return exec_graph->Run(exec_loss, exec_fetches, exec_feed_dict, num_micro_batches, 
                         cur_strategy_id, run_level);
}

} // namespace graph
} // namespace hetu
