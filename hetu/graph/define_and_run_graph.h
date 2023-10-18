#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/init/initializer.h"

namespace hetu {
namespace graph {

class DefineAndRunGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  DefineAndRunGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity) {
    std::srand(std::time(0));
    _op_to_exec_op_mapping.reserve(init_capacity);
    _tensor_to_exec_tensor_mapping.reserve(init_capacity);
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

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void Instantiate();

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor) override;

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
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
};

} // namespace graph
} // namespace hetu
