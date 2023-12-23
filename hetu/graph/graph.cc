#include "hetu/graph/headers.h"
#include "hetu/graph/define_by_run_graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/impl/communication/comm_group.h"
#include <thread>

namespace hetu {
namespace graph {

std::once_flag Graph::_init_flag;
std::vector<std::shared_ptr<Graph>> Graph::_global_graphs;
std::unordered_map<GraphName, std::shared_ptr<Graph>> Graph::_name_to_graphs;
std::shared_ptr<Graph> Graph::_default_eager_graph;
std::shared_ptr<Graph> Graph::_default_define_by_run_graph;
std::shared_ptr<Graph> Graph::_default_define_and_run_graph;
thread_local std::stack<GraphId> Graph::_cur_graph_ctx;

GraphId Graph::_next_graph_id() {
  static std::atomic<GraphId> _global_graph_id{0};
  return _global_graph_id++;
}

void Graph::Init() {
  // exit handler
  auto status = std::atexit([]() {
    HT_LOG_DEBUG << "Clearing and destructing all graphs...";
    Graph::_name_to_graphs.clear();
    Graph::_default_eager_graph = nullptr;
    Graph::_default_define_by_run_graph = nullptr;
    Graph::_default_define_and_run_graph = nullptr;
    for (auto& graph : Graph::_global_graphs) {
      if (graph == nullptr)
        continue;
      graph->Clear();
    }
    Graph::_global_graphs.clear();
    HT_LOG_DEBUG << "Destructed all graphs";
  });
  HT_ASSERT(status == 0)
      << "Failed to register the exit function for memory pools.";

  auto concurrency = std::thread::hardware_concurrency();
  Graph::_global_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_name_to_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_default_eager_graph =
    Graph::_make_new_graph<EagerGraph>("default_eager");
  Graph::_default_define_by_run_graph =
    Graph::_make_new_graph<DefineByRunGraph>("default_define_by_run");
  Graph::_default_define_and_run_graph =
    Graph::_make_new_graph<DefineAndRunGraph>("default_define_and_run");
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta) {
  Graph::InitOnce();
  if (inputs.empty() && op_meta.extra_deps.empty()) {
    HT_VALUE_ERROR_IF(Graph::_cur_graph_ctx.empty())
      << "The target graph must be explicitly passed or enqueued to ctx "
      << "when making a new op with zero inputs";
    HT_LOG_TRACE << "Make variable op on a " << Graph::GetGraph(Graph::_cur_graph_ctx.top()).type() << " graph";
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(Graph::_cur_graph_ctx.top()));
  } else {
    GraphId target_graph_id = std::numeric_limits<GraphId>::max();
    bool input_graph_changed = false;
    auto find_target_graph = [&](const Tensor& input) mutable {
      auto& input_graph = Graph::GetGraph(input->graph_id());
      if (target_graph_id == std::numeric_limits<GraphId>::max()) {
        target_graph_id = input->graph_id();
      } else if (target_graph_id != input->graph_id()) {
        input_graph_changed = true;
        if (input_graph.type() == GraphType::DEFINE_BY_RUN) {
          target_graph_id = input->graph_id();
        }
      }
      HT_VALUE_ERROR_IF(input_graph_changed &&
                        input_graph.type() != GraphType::EAGER &&
                        input_graph.type() != GraphType::DEFINE_BY_RUN && 
                        input_graph.type() != GraphType::DEFINE_AND_RUN)
        << "The target graph must be explicitly passed "
        << "when making new op to a " << input_graph.type() << " graph";
    };
    for (auto& input : inputs)
      find_target_graph(input);
    for (auto& in_dep : op_meta.extra_deps)
      find_target_graph(in_dep);
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(target_graph_id));
  }
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta, Graph& graph) {
  Graph::InitOnce();
  if (body->require_contig_inputs()) {
    for (auto& input : inputs) {
      if (!input->is_contiguous()) {
        auto op_id = input->get_contiguous_op_id();
        if (op_id.has_value()) {
          HT_LOG_TRACE << "Tensor " << input->name()
                       << " is not contiguous for op " << body->type()
                       << ". But it may have a contiguous copy, use it instead";
          // NOTE: Contiguous copy is created in the same graph as input.
          auto op = graph.GetOp(op_id.value());
          input = op->output(0);
        } else {
          HT_LOG_TRACE << "Make Contiguous op for Tensor " << input->name()
                       << " while making " << body->type() << " op";
          input = MakeContiguousOp(input);
        }
      }
    }
  }
  return graph.MakeOpInner(std::move(body), std::move(inputs),
                           std::move(op_meta));
}

TensorList Graph::Gradients(const TensorList& ys, const TensorList& xs,
                            const TensorList& grad_ys, int32_t num_ops_hint) {
  for (const auto& y : ys) {
    HT_VALUE_ERROR_IF(!y.is_defined()) << "Passed an undefined tensor";
    HT_VALUE_ERROR_IF(y->is_out_dep_linker())
      << "Cannot compute the gradient of " << y << " with operator "
      << y->producer()->type();
  }

  // Fill gradients
  TensorList filled_grads;
  filled_grads.reserve(ys.size());
  if (grad_ys.empty()) {
    for (const auto& y : ys) {
      // TODO: check whether requires grad
      filled_grads.emplace_back(MakeOnesLikeOp(y));
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(y->producer()->id());
    }
  } else {
    HT_VALUE_ERROR_IF(ys.size() != grad_ys.size())
      << "Provided " << grad_ys.size() << " gradients for " << ys.size()
      << " tensors";
    for (size_t i = 0; i < ys.size(); i++) {
      if (!grad_ys[i].is_defined()) {
        filled_grads.emplace_back(MakeOnesLikeOp(ys[i]));
      } else {
        filled_grads.push_back(grad_ys[i]);
      }
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(ys[i]->producer()->id());      
    }
  }

  Tensor2TensorListMap tensor_to_grads;
  Tensor2TensorMap tensor_to_reduced_grad;
  for (size_t i = 0; i < ys.size(); i++) {
    auto it = tensor_to_grads.find(ys[i]->id());
    if (it == tensor_to_grads.end())
      tensor_to_grads[ys[i]->id()] = {filled_grads[i]};
    else
      it->second.push_back(filled_grads[i]);
  }

  auto reduce_grad = [](const OpId& fw_op_id, const TensorList& unreduced_grads) -> Tensor {
    TensorList filtered;
    filtered.reserve(unreduced_grads.size());
    for (const auto& grad : unreduced_grads)
      if (grad.is_defined())
        filtered.push_back(grad);
    if (filtered.empty()) {
      return Tensor();
    } else if (filtered.size() == 1) {
      return filtered.front();
    } else {
      // Question: How to set op_meta properly?
      // if grad in filtered are all allreduce/reduce-scatter
      bool is_all_allreduce = true;
      bool is_all_reduce_scatter = true;
      for (const auto& grad : filtered) {
        if (is_comm_op(grad->producer())) {
          auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(grad->producer()->body());
          uint64_t comm_type = comm_op_impl.get_comm_type(grad->producer());
          if (comm_type != ALL_REDUCE_OP) {
            is_all_allreduce = false;
          }
          if (comm_type != REDUCE_SCATTER_OP) {
            is_all_reduce_scatter = false;
          }
          if (!is_all_allreduce && !is_all_reduce_scatter) {
            break;
          }
        } else {
          is_all_allreduce = false;
          is_all_reduce_scatter = false;
          break;
        }
      }
      Tensor grad_sum;
      if (is_all_allreduce || is_all_reduce_scatter) {
        TensorList partial_grad_list;
        for (const auto& grad : filtered) {
          Tensor partial_grad = grad->producer()->input(0);
          partial_grad_list.push_back(partial_grad);
        }
        // if allreduce/reduce-scatter group is different between input grads,
        // then assert error in state deduce process.
        Tensor partial_grad_sum = MakeSumOp(partial_grad_list, OpMeta().set_name("sum_op_for_partial_grad"));
        partial_grad_sum->set_is_grad(true);
        partial_grad_sum->producer()->set_fw_op_id(fw_op_id);
        DistributedStates ds_dst = filtered[0]->get_distributed_states();
        grad_sum = MakeCommOp(partial_grad_sum, ds_dst, OpMeta().set_name("comm_op_after_partial_grad_sum"));
      } else {
        grad_sum = MakeSumOp(filtered);
      }
      grad_sum->set_is_grad(true);
      grad_sum->producer()->set_fw_op_id(fw_op_id);
      return grad_sum;
    }
  };

  // traverse the forward graph in the reversed topo order
  auto topo = Graph::TopoSort(ys, num_ops_hint);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {  
    auto& op = it->get();
    TensorList grad_outputs;
    if (op->num_outputs() > 0) {
      grad_outputs.reserve(op->num_outputs());
      for (auto& output : op->outputs()) {
        auto grad = reduce_grad(op->id(), tensor_to_grads[output->id()]);
        tensor_to_reduced_grad[output->id()] = grad;
        grad_outputs.push_back(grad);
      }
    }
    if (op->num_inputs() > 0) {
      auto grad_inputs = op->Gradient(grad_outputs);
      for (size_t i = 0; i < op->num_inputs(); i++) {
        if (!grad_inputs[i].is_defined())
          continue;
        
        grad_inputs[i]->set_is_grad(true);
        grad_inputs[i]->producer()->set_fw_op_id(op->id());

        // states deduce
        const auto& grad_op = grad_inputs[i]->producer();
        const auto& ds_grad = grad_inputs[i]->get_distributed_states();
        Tensor final_grad = grad_inputs[i];
        if (ds_grad.is_valid()) {
          // HT_LOG_DEBUG << local_device << ": " << "grad_op: " << grad_op << ": states: " << ds_grad.ds_info() << ", shape: " << grad_inputs[i]->shape();
          if (ds_grad.get_dim(-2) > 1) { // partial->duplicate to sync the gradients for dp
            int32_t device_num = ds_grad.get_device_num();
            // std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
            std::pair<std::vector<int32_t>, int32_t> src2dst;
            if (is_variable_op(op->input(i)->producer()) && op->input(i)->get_distributed_states().zero()) {
              // attention: the result tensor was dp grouped split0, not really split0!
              // so should do allgather still within the same dp group later! 
              src2dst = {{-2}, 0}; // reduce-scatter
            } else {
              src2dst = {{-2}, -1}; // allreduce
            }
            std::unordered_map<int32_t, int32_t> res_states = ds_grad.combine_states(src2dst);
            std::vector<int32_t> res_order = ds_grad.combine_order(src2dst);
            DistributedStates ds_dst({device_num, res_states, res_order});
            HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " 
              << "backward: partial to duplicate: " << grad_inputs[i] 
              << ", dst states: " << ds_dst.ds_info();
            final_grad = MakeCommOp(grad_inputs[i], ds_dst, 
              OpMeta().set_name("comm_op_after_" + grad_op->name())); // allreduce
            final_grad->set_is_grad(true);
            final_grad->producer()->set_fw_op_id(op->id());
            // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": grad for tensor " << op->input(i) 
            //             << ", tensor shape = " << op->input(i)->shape() 
            //             << ", grad shape = " << final_grad->shape() 
            //             << ", grad src = " << grad_inputs[i]
            //             << ", grad src ds = " << grad_inputs[i]->get_distributed_states().ds_info()
            //             << ", grad dst ds = " << final_grad->get_distributed_states().ds_info()
            //             << ", comm type = " << reinterpret_cast<CommOpImpl&>(final_grad->producer()->body()).get_comm_type(final_grad->producer());
          }
        } 

        auto input = op->input(i);
        auto it = tensor_to_grads.find(input->id());
        if (it == tensor_to_grads.end())
          tensor_to_grads[input->id()] = {final_grad};
        else
          it->second.push_back(final_grad);
      }
    }

  }

  TensorList ret;
  ret.reserve(xs.size());
  for (auto& x : xs) {
    auto it = tensor_to_reduced_grad.find(x->id());
    if (it != tensor_to_reduced_grad.end())
      ret.push_back(it->second);
    else
      ret.emplace_back(Tensor());
  }
  return ret;
}

std::string GraphType2Str(GraphType type) {
  if (type == GraphType::EAGER) {
    return "eager";
  } else if (type == GraphType::DEFINE_BY_RUN) {
    return "define_by_run";
  } else if (type == GraphType::DEFINE_AND_RUN) {
    return "define_and_run";
  } else if (type == GraphType::EXECUTABLE) {
    return "executable";
  } else {
    HT_VALUE_ERROR << "Unrecognizable graph type: " << static_cast<int>(type);
    __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, GraphType type) {
  os << GraphType2Str(type);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Graph& graph) {
  os << "graph(name=" << graph.name() << ", id=" << graph.id()
     << ", type=" << graph.type() << ")";
  return os;
}

} // namespace graph
} // namespace hetu
