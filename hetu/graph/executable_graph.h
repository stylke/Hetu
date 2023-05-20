#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/group.h"

namespace hetu {
namespace graph {

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

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {});

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

 protected:
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

  NDArray& AllocVariableDataInner(
    const Tensor& tensor,
    const Initializer& init = VoidifiedInitializer()) override;

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
};

} // namespace graph
} // namespace hetu
