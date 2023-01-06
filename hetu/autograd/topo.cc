#include "hetu/autograd/topo.h"
#include "hetu/autograd/ops/Optimizer.h"
#include "hetu/autograd/ops/Communicate.h"
#include <queue>

namespace hetu {
namespace autograd {

using OpCRefQueue = std::queue<std::reference_wrapper<const Operator>>;

OpList TopoSort(const OpList& nodes, bool connect_p2p, bool skip_computed) {
  TIK(fn);
  OpList topo_order;
  std::unordered_map<OpId, int32_t> in_degrees;
  OpCRefQueue topo_sort_queue;
  OpCRefQueue traverse_queue;
  std::set<OpId> visited;
  // traverse all nodes that are connected with the target nodes
  // and enqueue nodes with zero in degrees into the topo sort queue
  auto traverse_fn = [&](const Operator& node) -> void {
    if (visited.find(node->id()) == visited.end()) {
      in_degrees[node->id()] =
        (skip_computed && node->is_computed()) ? 0 : node->in_degrees();
      if (in_degrees[node->id()] == 0)
        topo_sort_queue.push(node);
      traverse_queue.push(node);
      visited.insert(node->id());
    }
  };

  for (const Operator& node : nodes)
    traverse_fn(node);
  while (!traverse_queue.empty()) {
    const Operator& node = traverse_queue.front().get();
    traverse_queue.pop();
    OpRefList in_nodes_refs = node->input_ops_ref();
    for (const auto& in_node_ref : in_nodes_refs)
      traverse_fn(in_node_ref);
    if (connect_p2p && is_peer_to_peer_recv_op(node)) {
      auto& recv_op = reinterpret_cast<const P2PRecvOp&>(node);
      if (recv_op->is_distributed_tensor_recv_op()) { // distributed recv op
        for (auto& linked_op : recv_op->linked_ops()) {
          traverse_fn(linked_op);
        }
      } else { // pipeline recv op
        const auto& send_node = reinterpret_cast<const Operator&>(recv_op->send_op());
        traverse_fn(send_node);
      }
    }
  }

  // iteratively find the topo order
  while (!topo_sort_queue.empty()) {
    const Operator& node = topo_sort_queue.front().get();
    topo_sort_queue.pop();

    if (is_peer_to_peer_send_op(node) || is_peer_to_peer_recv_op(node)) {
      OpList local_send_recv_topo;
      if (is_peer_to_peer_send_op(node)) {
        auto& send_op = reinterpret_cast<const P2PSendOp&>(node);
        if (send_op->is_distributed_tensor_send_op()) {
          local_send_recv_topo = send_op->local_send_recv_topo();
        }
      }      
      if (is_peer_to_peer_recv_op(node)) {
        auto& recv_op = reinterpret_cast<const P2PRecvOp&>(node);
        if (recv_op->is_distributed_tensor_recv_op()) {
          local_send_recv_topo = recv_op->local_send_recv_topo();
        }
      }
      bool move_op_after = false;
      // local_topo在node前的那些send/recv op需要先执行完(已经出现在topo_order里)
      for (int32_t i = 0; i < local_send_recv_topo.size(); i++) {
        if (local_send_recv_topo[i]->id() == node->id()) {
          break;
        }
        auto it = find_if(topo_order.begin(), topo_order.end(),
        [&](Operator& op) -> bool {return op->id() == local_send_recv_topo[i]->id(); });
        if (it == topo_order.end()) {
          move_op_after = true;
          break;
        }
      }
      if (move_op_after) {
        topo_sort_queue.push(node);
        continue;
      }
    }
    
    if (!skip_computed || !node->is_computed())
      topo_order.push_back(node);
    OpRefList out_nodes_refs = node->output_ops_ref();
    for (const auto& out_node_ref : out_nodes_refs) {
      const Operator& out_node = out_node_ref.get();
      OpId out_node_id = out_node->id();
      if (visited.find(out_node_id) == visited.end())
        continue;
      in_degrees[out_node_id]--;
      if (in_degrees[out_node_id] == 0)
        topo_sort_queue.push(out_node);
    }
  }

  // ensure update ops are executed later
  for (size_t i = 0; i < topo_order.size(); i++) {
    if (is_optimizer_update_op(topo_order[i])) {
      Operator& update_op = topo_order[i];
      TensorId update_var_id = update_op->input(0)->id();
      for (size_t j = topo_order.size() - 1; j > i; j--) {
        if (is_optimizer_update_op(topo_order[j]))
          continue;
        auto it = std::find_if(
          topo_order[j]->inputs().begin(), topo_order[j]->inputs().end(),
          [&](const Tensor& edge) { return edge->id() == update_var_id; });
        if (it == topo_order[j]->inputs().end())
          continue;
        // insert topo_order[i] after topo_order[j]
        for (size_t k = i; k < j; k++)
          topo_order[k] = topo_order[k + 1];
        topo_order[j] = update_op;
        break;
      }
    }
  }

  TOK(fn);
  HT_LOG_TRACE << "Topo sort cost " << COST_MICROSEC(fn) << " microseconds";
  return topo_order;
}

OpList ExtendSubgraphWithCommunicationNodes(const OpList& nodes) {
  TIK(fn);
  OpList connected;
  OpCRefQueue traverse_queue;
  std::set<OpId> visited;

  auto traverse_fn = [&](const Operator& node) -> void {
    if (visited.find(node->id()) == visited.end()) {
      traverse_queue.push(node);
      visited.insert(node->id());
    }
  };

  for (const Operator& node : nodes)
    traverse_fn(node);
  while (!traverse_queue.empty()) {
    const Operator& node = traverse_queue.front().get();
    traverse_queue.pop();
    connected.push_back(node);
    OpRefList in_nodes_refs = node->input_ops_ref();
    for (const auto& node_ref : in_nodes_refs)
      if (is_communucation_op(node_ref))
        traverse_fn(node_ref);
    OpRefList out_nodes_refs = node->output_ops_ref();
    for (const auto& node_ref : out_nodes_refs) 
      if (is_communucation_op(node_ref))
        traverse_fn(node_ref);
  }

  TOK(fn);
  HT_LOG_TRACE << "Extend subgraph with communication nodes cost "
               << COST_MICROSEC(fn) << " microseconds";
  return connected;
}

std::tuple<OpList, OpList> disentangle_forward_and_backward_nodes(
  const OpList& nodes, const TensorList& losses, bool connect_p2p) {
  // traverse forward nodes (including losses)
  OpCRefQueue traverse_queue;
  for (const Tensor& loss : losses)
    traverse_queue.push(loss->producer());
  std::set<OpId> fw_set;
  while (!traverse_queue.empty()) {
    const Operator& node = traverse_queue.front().get();
    traverse_queue.pop();
    fw_set.insert(node->id());
    OpRefList in_nodes_refs = node->input_ops_ref();
    for (auto& in_node_ref : in_nodes_refs) {
      const Operator& in_node = in_node_ref.get();
      if (fw_set.find(in_node->id()) == fw_set.end())
        traverse_queue.push(in_node);
    }
    if (connect_p2p && is_peer_to_peer_recv_op(node)) {
      const auto& send_node = reinterpret_cast<const Operator&>(
        reinterpret_cast<const P2PRecvOp&>(node)->send_op());
      if (fw_set.find(send_node->id()) == fw_set.end()) {
        traverse_queue.push(send_node);
      }
    }
  }

  // get the forward nodes
  OpList fw_nodes;
  fw_nodes.reserve(fw_set.size());
  std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(fw_nodes),
               [&fw_set](const Operator& node) {
                 return fw_set.find(node->id()) != fw_set.end();
               });

  // get the backward nodes
  OpList bw_nodes;
  bw_nodes.reserve(nodes.size() - fw_nodes.size());
  std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(bw_nodes),
               [&fw_set](const Operator& node) {
                 return fw_set.find(node->id()) == fw_set.end();
               });

  return {fw_nodes, bw_nodes};
}

} // namespace autograd
} // namespace hetu
