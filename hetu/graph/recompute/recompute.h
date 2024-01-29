#pragma once

#include "hetu/graph/common.h"
#include "hetu/graph/define_and_run_graph.h"

namespace hetu {
namespace graph {

class Recompute {
 public:
  static bool enabled() {
    return _enabled;
  }

  static void set_recompute_enabled() {
    _enabled = true;
  }

  static void reset_recompute_enabled() {
    _enabled = false;
  }

  static void InsertRecomputedOps(const OpRefList& topo_order);

 protected:
  static void GetMaxRecomputeSubGraph(Op2OpRefMap& recompute_subgraph, bool get_inputs, bool get_outputs);

  static Operator& DuplicateRecomputedOp(const Operator& origin_op, const Op2OpRefMap& filtered_recomputed_ops,
                                         const TensorList& first_mapped_grad_inputs, Op2OpMap& origin_to_recomputed_map);

  static bool _enabled;
};

} // namespace graph
} // namespace hetu