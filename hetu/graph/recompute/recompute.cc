#include "hetu/graph/recompute/recompute.h"

namespace hetu {
namespace graph {

bool Recompute::_enabled = false;

// helper functions
namespace {
  inline int get_out_idx_of(const Operator& op, const Tensor& tensor) {
    auto num_output = op->num_outputs();
    for (int out_pos = 0; out_pos < num_output; out_pos++) {
      if (op->output(out_pos)->id() == tensor->id()) {
        return out_pos;
      }
    }
    return -1;
  }
} // namespace

void Recompute::GetMaxRecomputeSubGraph(Op2OpRefMap& recompute_subgraph, bool get_inputs, bool get_outputs) {
  OpRefQueue to_visit;
  for (auto& op_opref : recompute_subgraph) {
    to_visit.push(op_opref.second);
  }
  recompute_subgraph.clear();

  while (!to_visit.empty()) {
    auto& op_ref = to_visit.front();
    to_visit.pop();
    if (recompute_subgraph.find(op_ref.get()->id()) != recompute_subgraph.end()) {
      continue;
    }
    recompute_subgraph.insert({op_ref.get()->id(), op_ref});
    // add recomputed inputs to recompute subgraph
    if (get_inputs) {
      auto& op_inputs = op_ref.get()->inputs();
      for (auto& input : op_inputs) {
        auto& op = input->producer();
        if (op->op_meta().is_recompute && 
            recompute_subgraph.find(op->id()) == recompute_subgraph.end()) {
          to_visit.push(std::ref(op));
        }
      }
    }
    // add recomputed outputs to recompute subgraph
    if (get_outputs) {
      auto& op_outputs = op_ref.get()->outputs();
      for (auto& output : op_outputs) {
        auto& out_consumers = output->consumers();
        for (auto& op_ref : out_consumers) {
          auto& op = op_ref.get();
          if (op->op_meta().is_recompute && 
              recompute_subgraph.find(op->id()) == recompute_subgraph.end()) {
            to_visit.push(op_ref);
          }
        }
      }
    }
  }
}

Operator& Recompute::DuplicateRecomputedOp(const Operator& origin_op, const Op2OpRefMap& filtered_recomputed_ops,
                                           const TensorList& first_mapped_grad_inputs, Op2OpMap& origin_to_recomputed_map) {
  auto iter = origin_to_recomputed_map.find(origin_op->id());
  if (iter != origin_to_recomputed_map.end()) {
    return iter->second;
  }
  TensorList new_inputs;
  bool has_recomputed_input = false;
  auto& origin_inputs = origin_op->inputs();
  for (auto& input : origin_inputs) {
    auto& input_op = input->producer();
    if (filtered_recomputed_ops.find(input_op->id()) == filtered_recomputed_ops.end()) {
      new_inputs.push_back(input);
    } else {
      has_recomputed_input = true;
      // find the output position of input in input_op
      auto out_idx = get_out_idx_of(input_op, input);
      HT_ASSERT(out_idx != -1)
        << "Cannot find input " << input << " in the outputs of recomputed op " << input_op;
      auto& new_input = DuplicateRecomputedOp(input_op, filtered_recomputed_ops,
                                              first_mapped_grad_inputs, origin_to_recomputed_map);
      new_inputs.push_back(new_input->output(out_idx));
    }
  }
  auto new_op_meta = OpMeta().set(origin_op->op_meta())
                             .set_is_recompute(false)
                             .set_name(origin_op->name() + "_recompute");
  // add the execution dependency
  if (!has_recomputed_input) {
    new_op_meta.set_extra_deps(first_mapped_grad_inputs);
  }
  auto& new_op = Graph::MakeOp(origin_op->_body, std::move(new_inputs), std::move(new_op_meta));
  origin_to_recomputed_map.insert({origin_op->id(), new_op});
  return origin_to_recomputed_map[origin_op->id()];
}

void Recompute::InsertRecomputedOps(const OpRefList& topo_order) {
  auto is_grad_op = [](const OpRef& op_ref) {
    return Operator::all_output_tensors_of(op_ref.get(), 
      [&](const Tensor& tensor) -> bool {
        return tensor->is_grad();
      });
  };

  // Find candidate recomputed ops.
  auto has_grad_consumer = [is_grad_op](const Tensor& tensor) {
    return Tensor::any_consumer_of(tensor, [is_grad_op](const OpRef& op_ref) -> bool {
      return is_grad_op(op_ref);
    });
  };
  HT_LOG_DEBUG << "[Recompute] find candidate recomputed ops begin...";
  OpRefList candidate_recomputed_ops;
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    if (!op->op_meta().is_recompute) {
      continue;
    }
    // No recomputation if there is no grad op in the outputs
    if (Operator::any_output_tensor_of(op, has_grad_consumer)) {
      candidate_recomputed_ops.push_back(op_ref);
    }
  }
  // Iterate over all candidate recomputed ops and
  // construct recompute subgraph with continuous recomputed ops.
  OpIdSet visited_ops;
  for (auto& candidate_recomputed_op_ref : candidate_recomputed_ops) {
    auto& candidate_recomputed_op = candidate_recomputed_op_ref.get();
    if (visited_ops.find(candidate_recomputed_op->id()) != visited_ops.end()) {
      continue;
    }
    // Get max continuous recompute subgraph first.
    HT_LOG_DEBUG << "[Recompute] get max recomputed subgraph begin...";
    Op2OpRefMap max_recompute_subgraph = {{candidate_recomputed_op->id(), candidate_recomputed_op_ref}};
    GetMaxRecomputeSubGraph(max_recompute_subgraph, true, true);
    OpIdList max_recompute_subgraph_ids;
    for (auto& op : max_recompute_subgraph) {
      max_recompute_subgraph_ids.push_back(op.first);
    }
    visited_ops.insert(max_recompute_subgraph_ids.begin(), max_recompute_subgraph_ids.end());
    // Filter recomputed ops that directly output to grad ops.
    HT_LOG_DEBUG << "[Recompute] filter recomputed ops that directly output to grad ops begin...";
    Op2OpRefMap filtered_recomputed_ops;
    Op2OpRefMap mapped_grad_ops;
    for (auto& op_opref : max_recompute_subgraph) {
      bool inserted = false;
      auto op_id = op_opref.first;
      auto& op_ref = op_opref.second;
      auto& op_outputs = op_ref.get()->outputs();
      for (auto& output : op_outputs) {
        auto& out_consumers = output->consumers();
        for (auto& out_op_ref : out_consumers) {
          if (!is_grad_op(out_op_ref)) {
            continue;
          }
          mapped_grad_ops.insert({out_op_ref.get()->id(), out_op_ref});
          if (!inserted) {
            inserted = true;
            if (filtered_recomputed_ops.find(op_ref.get()->id()) != filtered_recomputed_ops.end()) {
              continue;
            }
            filtered_recomputed_ops.insert({op_id, op_ref});
          }
        }
      }
    }
    // Get inputs of filtered recomputed ops which eventually output to grad ops.
    HT_LOG_DEBUG << "[Recompute] get inputs of filtered recomputed ops which eventually output to grad ops begin...";
    GetMaxRecomputeSubGraph(filtered_recomputed_ops, true, false);
    // Find inputs of the first mapped grad op in the topo order.
    HT_LOG_DEBUG << "[Recompute] find inputs of the first mapped grad op in the topo order begin...";
    TensorList first_mapped_grad_inputs;
    for (auto& op_ref : topo_order) {
      if (mapped_grad_ops.find(op_ref.get()->id()) == mapped_grad_ops.end()) {
        continue;
      }
      auto& op_inputs = op_ref.get()->inputs();
      for (auto& input : op_inputs) {
        auto& op = input->producer();
        // only insert inputs that are not recomputed
        if (filtered_recomputed_ops.find(op->id()) == filtered_recomputed_ops.end()) {
          first_mapped_grad_inputs.push_back(input);
        }
      }
      if (!first_mapped_grad_inputs.empty()) {
        break;
      }
    }
    // Duplicate recomputed ops
    HT_LOG_DEBUG << "[Recompute] duplicate recomputed ops begin...";
    Op2OpMap origin_to_recomputed_map;
    for (auto& op_opref : mapped_grad_ops) {
      auto& mapped_grad_op = op_opref.second.get();
      auto num_inputs = mapped_grad_op->num_inputs();
      for (int i = 0; i < num_inputs; i++) {
        auto& input = mapped_grad_op->input(i);
        auto& input_op = input->producer();
        if (filtered_recomputed_ops.find(input_op->id()) != filtered_recomputed_ops.end()) {
          auto out_idx = get_out_idx_of(input_op, input);
          HT_ASSERT(out_idx != -1)
            << "Cannot find input " << input << " in the outputs of recomputed op " << input_op;
          auto& recomputed_op = DuplicateRecomputedOp(input_op, filtered_recomputed_ops,
                                                      first_mapped_grad_inputs, origin_to_recomputed_map);
          HT_LOG_DEBUG << "[Recompute] replacing mapped_grad_op " << mapped_grad_op << " input[" << i
                      << "] with recomputed_op " << recomputed_op << " output[" << out_idx << "]...";
          mapped_grad_op.get()->replace_input(i, recomputed_op->output(out_idx));
        }
      }
    }
  }
}

} // namespace graph
} // namespace hetu