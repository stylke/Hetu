#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/group.h"

namespace hetu {
namespace graph {

struct ExecutePlan {
  OpRefList local_fw_topo;
  OpRefList local_bw_topo;
  OpRefList local_topo;
  TensorIdSet shared_weight_tensor;
  OpIdSet shared_weight_p2p;
  TensorIdSet accumulated_tensor;
  OpIdSet accumulated_ops;

  void update(OpRefList& _local_fw_topo, OpRefList& _local_bw_topo, 
              OpRefList& _local_topo, TensorIdSet& _shared_weight_tensor,
              OpIdSet& _shared_weight_p2p, TensorIdSet& _accumulated_tensor, 
              OpIdSet& _accumulated_ops) {
    local_fw_topo = _local_fw_topo;
    local_bw_topo = _local_bw_topo;
    local_topo = _local_topo;
    shared_weight_tensor = _shared_weight_tensor;
    shared_weight_p2p = _shared_weight_p2p;
    accumulated_tensor = _accumulated_tensor;
    accumulated_ops = _accumulated_ops;
  }
};

class ExecutableGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  ExecutableGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity) {}

 public:
  ExecutableGraph(const constrcutor_access_key&, GraphName name,
                  size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : ExecutableGraph(name, init_capacity) {}

  // bool MapOpsToParallelDevices(const DeviceGroup& placement_group);

  bool Instantiate(const TensorList& fetches, const Device& placement);

  NDArrayList Run(const TensorList& fetches, 
                  const FeedDict& feed_dict = {});

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const int num_micro_batches = 1);

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

  void SetStages(const std::vector<DeviceGroup>& device_groups) {
    _stages = device_groups;
  }

  void SetShapePlan(int num) {
    _active_plan = num;
  }

  void InitShapePlan(Tensor2ShapeMap&& shape_plan) {
    _shape_plan_pools.clear();
    _shape_plan_pools.push_back(std::move(shape_plan));
  }

  void AddShapePlan(Tensor2ShapeMap&& shape_plan) {
    _shape_plan_pools.push_back(std::move(shape_plan));
  }

  void RecordTensorShape(const TensorId& key, const HTShape& value) {
    auto& shape_plan = _shape_plan_pools[_active_plan];
    auto it = shape_plan.find(key);
    if (it != shape_plan.end()) {
      // already existed, then must be equal
      HT_ASSERT(it->second.size() == value.size())
        << "Tensor " << key << " is already exited in shape plan but is unequal";
      for (size_t i = 0; i < value.size(); i++) { 
        HT_ASSERT(it->second[i] == value[i])
          << "Tensor " << key << " is already exited in shape plan but is unequal";
      }
      return;
    }
    shape_plan.insert(std::make_pair(key, value));
  }

  void RecordTensorShape(const TensorId& key, HTShape&& value) {
    auto& shape_plan = _shape_plan_pools[_active_plan];
    auto it = shape_plan.find(key);
    if (it != shape_plan.end()) {
      // already existed, then must be equal
      HT_ASSERT(it->second.size() == value.size())
        << "Tensor " << key << " is already existed in shape plan but is unequal";
      for (size_t i = 0; i < value.size(); i++) { 
        HT_ASSERT(it->second[i] == value[i])
          << "Tensor " << key << " is already existed in shape plan but is unequal";
      }
      return;
    } 
    shape_plan.insert(std::make_pair(key, std::move(value)));
  }

  const HTShape& GetTensorShape(const TensorId& key) const {
    auto& shape_plan = _shape_plan_pools[_active_plan];
    auto it = shape_plan.find(key);
    HT_ASSERT(it != shape_plan.end())
      << "Tensor " << key << " is not existed in current shape plan";
    return it->second;
  }

 protected:
  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
  GenerateGpipeSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
  GeneratePipedreamFlushSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  void ComputeFunc(size_t& micro_batch_id, const OpRefList& topo, RuntimeContext& runtime_ctx, 
                  Tensor2NDArrayMap& tensor2data, Tensor2IntMap& tensor2degrees, 
                  Tensor2NDArrayMap& grad_accumulation, bool grad_accumulation_finished,
                  const TensorIdSet& shared_weight_tensor, const OpIdSet& shared_weight_p2p,
                  const TensorIdSet& accumulated_tensor, const OpIdSet& accumulated_ops,
                  const FeedDict& feed_dict, const TensorList& fetches,
                  const std::unordered_map<TensorId, size_t>& fetch_indices, 
                  bool& is_continuous_p2p);

  void SubstituteCommOp(const OpRefList& topo_order);

  void CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                 std::unordered_map<int32_t, int32_t> split_target_state,
                 int32_t depth, bool need_split, int32_t& device_index, 
                 Operator& comm_op, TensorList& send_datas, 
                 std::vector<int32_t>& dsts, int32_t& used_device_index);

  Tensor CrossReceive(int32_t depth, int32_t& device_index, Operator& comm_op, 
                      TensorList& recv_datas, std::vector<int32_t>& srcs,
                      Tensor& self_send_data, int32_t& used_device_index);

  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray& GetVariableDataInner(const Tensor& tensor) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor) override;

  NDArray& AllocVariableDataInner(
    const Tensor& tensor,
    const Initializer& init = VoidifiedInitializer(),
    uint64_t seed = 0, const HTShape& global_shape = HTShape()) override;

  void RegisterVariableDataInner(
    const Tensor& tensor, NDArray data,
    const Initializer& init = VoidifiedInitializer()) override;

  void ReplaceInput(Operator& op, size_t input_index, Tensor& new_input) {
    auto& old_input = op->_inputs[input_index];
    old_input->DelConsumer(op);
    op->_inputs[input_index] = new_input;
    new_input->AddConsumer(op);
  }

  void AddInDeps(Operator& op, const TensorList& in_deps) {
    if (in_deps.empty()) {
      return;
    }
    if (in_deps.size() == 1) {
      op->_extra_in_dep_linkers.push_back(in_deps.front());
    } else {
      op->_extra_in_dep_linkers.push_back(
        MakeGroupOp(OpMeta()
                      .set_extra_deps(in_deps)
                      .set_name(op->name() + "_extra_in_dep")));
    }
    op->_extra_in_dep_linkers.back()->AddConsumer(op);
  }

  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  std::vector<DeviceGroup> _stages;
  int _num_micro_batches;
  ExecutePlan _execute_plan;
  size_t _active_plan;
  std::vector<Tensor2ShapeMap> _shape_plan_pools;
  std::vector<std::unique_ptr<Event>> _p2p_events;
};

} // namespace graph
} // namespace hetu
