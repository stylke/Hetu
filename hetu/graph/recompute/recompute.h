#pragma once

#include "hetu/graph/common.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/graph.h"
#include <functional>
#include <stack>

namespace hetu {
namespace graph {

class Recompute {
 public:
  static const std::vector<bool>& multi_recompute() {
    return _multi_recompute_stack.top();
  }

  static void push_recompute_enabled(const std::vector<bool>& multi_recompute) {
    _multi_recompute_stack.push(multi_recompute);
  }

  static void pop_recompute_enabled() {
    _multi_recompute_stack.pop();
  }

  static void InsertRecomputedOps(const OpRefList& topo_order);

 protected:
  static bool IsNoRecomputedOp(Operator& op);

  static void GetMaxRecomputeSubGraph(Op2OpRefMap& recompute_subgraph, bool get_inputs, bool get_outputs);

  static bool HasFilterOpInPath(const Operator& op, std::function<bool(const Operator &)> filter_fn,
                                std::unordered_map<OpId, bool>& has_filter_op_map);

  static Operator& DuplicateRecomputedOp(const Operator& origin_op, const Op2OpRefMap& filtered_recomputed_ops,
                                         const TensorList& first_mapped_grad_inputs, Op2OpMap& origin_to_recomputed_map,
                                         ExecutableGraph& cur_exec_graph);

  static std::stack<std::vector<bool>> _multi_recompute_stack;
};

} // namespace graph
} // namespace hetu