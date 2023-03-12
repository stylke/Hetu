#pragma once

#include "hetu/graph/graph.h"

namespace hetu {
namespace graph {

class EagerGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  EagerGraph(size_t init_capacity)
  : Graph(init_capacity), _runtime_ctxs(init_capacity) {
    _op_to_num_destructed_outputs.reserve(init_capacity);
  }

 public:
  EagerGraph(const constrcutor_access_key&,
             size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : EagerGraph(init_capacity) {}

  NDArrayList Run(const TensorList& fetches,
                  const Tensor2NDArrayMap& feed_dict = {});

  GraphType type() const {
    return GraphType::EAGER;
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  NDArray& GetOrCompute(Tensor& tensor);
  
  void RemoveTensor(const Tensor& tensor);

  void RemoveOp(Operator& op) override {
    _runtime_ctxs.remove(op->id());
    _op_to_num_destructed_outputs.erase(op->id());
    Graph::RemoveOp(op);
  }
  
  void Clear() override {
    _runtime_ctxs.clear();
    Graph::Clear();
  }

  RuntimeContext _runtime_ctxs;
  std::unordered_map<OpId, size_t> _op_to_num_destructed_outputs;
};

} // namespace graph
} // namespace hetu
