#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/init/initializer.h"

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

  bool MapOpsToParallelDevices(const DeviceGroup& placement_group);

  bool Instantiate(const TensorList& fetches, const Device& placement);

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {});

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

 protected:
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

  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
};

} // namespace graph
} // namespace hetu
