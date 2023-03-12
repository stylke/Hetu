#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"

namespace hetu {
namespace graph {

class DefineAndRunGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  DefineAndRunGraph(size_t init_capacity)
  : Graph(init_capacity), _instantiated(false) {
    _op_to_exec_op_mapping.reserve(init_capacity);
    _tensor_to_exec_tensor_mapping.reserve(init_capacity);
  }

 public:
  DefineAndRunGraph(const constrcutor_access_key&,
                    size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : DefineAndRunGraph(init_capacity) {}

  NDArrayList Run(const TensorList& fetches,
                  const Tensor2NDArrayMap& feed_dict = {});

  GraphType type() const {
    return GraphType::DEFINE_AND_RUN;
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void Instantiate();

  void RemoveOp(Operator& op) override {
    _op_to_exec_op_mapping.erase(op->id());
    Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
      _tensor_to_exec_tensor_mapping.erase(tensor->id());
    });
    Graph::RemoveOp(op);
  }

  void Clear() override {
    _op_to_exec_op_mapping.clear();
    _tensor_to_exec_tensor_mapping.clear();
    Graph::Clear();
  }

  std::shared_ptr<ExecutableGraph> _exec_graph;
  Op2OpMap _op_to_exec_op_mapping;
  Tensor2TensorMap _tensor_to_exec_tensor_mapping;
  bool _instantiated;
};

} // namespace graph
} // namespace hetu
