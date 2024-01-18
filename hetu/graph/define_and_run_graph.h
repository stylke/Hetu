#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/init/initializer.h"

namespace hetu {
namespace graph {

class ExecGraphPlan {
 public:
  std::shared_ptr<ExecutableGraph> exec_graph;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  TensorList fetches;

  ExecGraphPlan(const std::shared_ptr<ExecutableGraph>& _exec_graph, const Op2OpMap& _op_to_exec_op_mapping, 
                const Tensor2TensorMap& _tensor_to_exec_tensor_mapping)
  : exec_graph(_exec_graph), 
    op_to_exec_op_mapping(_op_to_exec_op_mapping),
    tensor_to_exec_tensor_mapping(_tensor_to_exec_tensor_mapping) {}
  
  ExecGraphPlan(std::shared_ptr<ExecutableGraph>&& _exec_graph, Op2OpMap&& _op_to_exec_op_mapping, 
                Tensor2TensorMap&& _tensor_to_exec_tensor_mapping)
  : exec_graph(std::move(_exec_graph)), 
    op_to_exec_op_mapping(std::move(_op_to_exec_op_mapping)),
    tensor_to_exec_tensor_mapping(std::move(_tensor_to_exec_tensor_mapping)) {}
};

class DefineAndRunGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;
  friend class SwitchExecGraph;

  DefineAndRunGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity),
    _init_capacity(init_capacity) {
    std::srand(std::time(0));
  }

 public:
  DefineAndRunGraph(const constrcutor_access_key&, GraphName name,
                    size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : DefineAndRunGraph(name, init_capacity) {}

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {});

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const int num_micro_batches = 1);

  GraphType type() const {
    return GraphType::DEFINE_AND_RUN;
  }

  const ExecGraphPlan& GetPlan(size_t num) const {
    HT_ASSERT(num < _exec_graph_plan_pool.size());
    return _exec_graph_plan_pool[num];
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void Instantiate(const OpRefList& topo,
                   Tensor2ShapeMap& shape_plan);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor) override;

  DeviceGroup GetVariableDeviceGroupInner(const Tensor& tensor) override;

  void RemoveOp(Operator& op) override {
    auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_plan].op_to_exec_op_mapping;
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_plan].tensor_to_exec_tensor_mapping;
    op_to_exec_op_mapping.erase(op->id());
    Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
      tensor_to_exec_tensor_mapping.erase(tensor->id());
    });
    Graph::RemoveOp(op);
  }

  void Clear() override {
    _add_on_inits.clear();
    _multi_device_groups.clear();
    _exec_graph_plan_pool.clear();
    _shape_plan_pool.clear();
    Graph::Clear();
  }
  
  void SetPlan(size_t num) {
    _active_plan = num;
    _is_active = true;
  }

  size_t _init_capacity;
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  std::vector<DeviceGroupList> _multi_device_groups; // all the device groups of ops, in the order of MakeOp calls

  std::vector<ExecGraphPlan> _exec_graph_plan_pool;
  std::vector<Tensor2ShapeMap> _shape_plan_pool;
  size_t _active_plan;
  bool _is_active = false;

 public: 
  /* utils for parallel plan changing test case */
  static void dp2tp(Operator& op);
  static void tp2dp(Operator& op);
  void SetVariableDistributedStates(Operator& op, int32_t dp, int32_t tp);
  void InstantiateTestCase(const OpRefList& topo,
                           Tensor2ShapeMap& shape_plan);
};

} // namespace graph
} // namespace hetu
