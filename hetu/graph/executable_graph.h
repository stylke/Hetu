#pragma once

#include "hetu/graph/graph.h"

namespace hetu {
namespace graph {

class ExecutableGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  ExecutableGraph(size_t init_capacity) : Graph(init_capacity) {}

 public:
  ExecutableGraph(const constrcutor_access_key&,
                  size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : ExecutableGraph(init_capacity) {}

  bool MapOpsToParallelDevices(const DeviceGroup& placement_group);

  bool Instantiate(const TensorList& fetches, const Device& placement);
  
  NDArrayList Run(const TensorList& fetches,
                  const Tensor2NDArrayMap& feed_dict = {});

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ReplaceInput(Operator& op, size_t input_index, Tensor& new_input) {
    auto& old_input = op->_inputs[input_index];
    old_input->DelConsumer(op);
    op->_inputs[input_index] = new_input;
    new_input->AddConsumer(op);
  }
};

} // namespace graph
} // namespace hetu
