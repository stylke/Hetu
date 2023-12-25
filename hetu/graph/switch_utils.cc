#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

// temp utils of changing ds
static DistributedStates dup2split(const DistributedStates& ds, int32_t split_dim) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "dup should >= 2 and could be divided by 2";
      new_states[-1] = kv.second / 2;
    } else if (kv.first == split_dim) {
      new_states[kv.first] = kv.second * 2;
    } else {
      HT_ASSERT(kv.second == 1) 
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "dup2split " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

static DistributedStates split2dup(const DistributedStates& ds, int32_t split_dim) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      new_states[-1] = kv.second * 2;
    } else if (kv.first == split_dim) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "split should >= 2 and could be divided by 2";
      new_states[kv.first] = kv.second / 2;
    } else {
      HT_ASSERT(kv.second == 1)
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "split2dup " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

static DistributedStates split2split(const DistributedStates& ds, int32_t split_dim_before, int32_t split_dim_after) {
  const auto& states = ds.get_states();
  std::unordered_map<int32_t, int32_t> new_states;
  for (const auto& kv : states) {
    if (kv.first == -2) {
      HT_ASSERT(kv.second == 1)
        << "partial shouldn't exist";
    } else if (kv.first == -1) {
      HT_ASSERT(kv.second == 1)
        << "dup shouldn't exist";
    } else if (kv.first == split_dim_before) {
      HT_ASSERT(kv.second >= 2 && kv.second % 2 == 0)
        << "before split should >= 2 and could be divided by 2";
      new_states[kv.first] = kv.second / 2;
    } else if (kv.first == split_dim_after) {
      new_states[kv.first] = kv.second * 2;
    } else {
      HT_ASSERT(kv.second == 1)
        << "there shouldn't be another split dim";
    }
  }
  HT_LOG_DEBUG << "split2split " << states << " to " << new_states;
  if (ds.get_placement_group().empty()) {
    return DistributedStates(ds.get_device_num(), new_states, ds.get_order());
  }
  return DistributedStates(ds.get_placement_group(), new_states, ds.get_order());
}

// dp /= 2 and tp *= 2
// Note only work when tp is already >= 2
// need to revamp that
void DefineAndRunGraph::dp2tp(Operator& op) {
  if (is_variable_op(op) && op->_body->type() == "ParallelVariableOp") {
    auto& variable = op->output(0);
    HT_ASSERT(variable->has_distributed_states())
      << "variable " << variable << " doesn't have distributed_states";
    const auto& ds = variable->get_distributed_states();
    // pure dup
    if (ds.check_pure_duplicate()) {
      // HT_LOG_INFO << "variable " << variable << " is pure dup, do not change";
      return;
    }
    // split 0, row parallel
    if (ds.get_dim(0) >= 2) {
      // HT_LOG_INFO << "variable " << variable << " is splited at dim 0, split it more";
      auto new_ds = dup2split(ds, 0);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    // split 1, col parallel
    if (ds.get_dim(1) >= 2) {
      // HT_LOG_INFO << "variable " << variable << " is splited at dim 1, split it more";
      auto new_ds = dup2split(ds, 1);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_placeholder_op(op)) {
    auto& placeholder = op->output(0);
    HT_ASSERT(placeholder->has_distributed_states())
      << "placeholder " << placeholder << " doesn't have distributed_states";
    const auto& ds = placeholder->get_distributed_states();
    // split 0 means dp for placeholder
    // input related
    if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
      // HT_LOG_INFO << "placeholder " << placeholder << " is splited at dim 0, split it less";
      placeholder->set_distributed_states(split2dup(ds, 0));
      return;
    } 
    // mask related
    if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
      // HT_LOG_INFO << "placeholder " << placeholder << " is splited at dim 0 and dim 1, split dim 0 less and dim 1 more";
      placeholder->set_distributed_states(split2split(ds, 0, 1));
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_comm_op(op)) {
    auto& comm_output = op->output(0);
    HT_ASSERT(comm_output->has_distributed_states())
      << "comm_output " << comm_output << " doesn't have distributed_states";
    const auto& ds = comm_output->get_distributed_states();
    // gradient allreduce
    if (op->name().find("comm_op_after_") != std::string::npos) {
      // pure dup
      if (ds.check_pure_duplicate()) {
        return;
      }
      // col parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // "partial_grad_sum" is wte_table
        if (op->name().find("partial_grad_sum") != std::string::npos || op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = dup2split(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = split2dup(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      } 
      // row parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(1) >= 2) {
        if (op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = dup2split(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = split2dup(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      }
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    } else {
      // comm in row parallel & last col parallel & grad comm in row parallel
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // HT_LOG_INFO << "comm_output " << comm_output << " is splited at dim 0, split it less";
        auto new_ds = split2dup(ds, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      // grad comm in last col parallel
      if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
        // HT_LOG_INFO << "comm_output " << comm_output << " is splited at dim 0 and dim 1, split dim 0 less and dim 1 more";
        auto new_ds = split2split(ds, 0, 1);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    }
  }
  HT_LOG_INFO << "op " << op << " is not a variable/placeholder nor a comm";
  HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
}

// dp *= 2 and tp /= 2
// Note only work when dp is already >= 2
// need to revamp that
void DefineAndRunGraph::tp2dp(Operator& op) {
  if (is_variable_op(op) && op->_body->type() == "ParallelVariableOp") {
    auto& variable = op->output(0);
    HT_ASSERT(variable->has_distributed_states())
      << "variable " << variable << " doesn't have distributed_states";
    const auto& ds = variable->get_distributed_states();
    // pure dup
    if (ds.check_pure_duplicate()) {
      // HT_LOG_INFO << "variable " << variable << " is pure dup, do not change";
      return;
    }
    // split 0, row parallel
    if (ds.get_dim(0) >= 2) {
      auto new_ds = split2dup(ds, 0);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    // split 1, col parallel
    if (ds.get_dim(1) >= 2) {
      auto new_ds = split2dup(ds, 1);
      (std::dynamic_pointer_cast<ParallelVariableOpImpl>(op->_body))->set_ds(new_ds);
      variable->set_distributed_states(new_ds);
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_placeholder_op(op)) {
    auto& placeholder = op->output(0);
    HT_ASSERT(placeholder->has_distributed_states())
      << "placeholder " << placeholder << " doesn't have distributed_states";
    const auto& ds = placeholder->get_distributed_states();
    // split 0 means dp for placeholder
    // input related
    if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
      placeholder->set_distributed_states(dup2split(ds, 0));
      return;
    } 
    // mask related
    if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
      placeholder->set_distributed_states(split2split(ds, 1, 0));
      return;
    }
    HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
  }
  if (is_comm_op(op)) {
    auto& comm_output = op->output(0);
    HT_ASSERT(comm_output->has_distributed_states())
      << "comm_output " << comm_output << " doesn't have distributed_states";
    const auto& ds = comm_output->get_distributed_states();
    // gradient allreduce
    if (op->name().find("comm_op_after_") != std::string::npos) {
      // pure dup
      if (ds.check_pure_duplicate()) {
        return;
      }
      // col parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        // "partial_grad_sum" is wte_table
        if (op->name().find("partial_grad_sum") != std::string::npos || op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = split2dup(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = dup2split(ds, 0);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      } 
      // row parallel gradient allreduce
      if (ds.get_dim(-1) >= 2 && ds.get_dim(1) >= 2) {
        if (op->name().find("weight") != std::string::npos || op->name().find("bias") != std::string::npos) {
          auto new_ds = split2dup(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        } else {
          auto new_ds = dup2split(ds, 1);
          (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
          comm_output->set_distributed_states(new_ds);
          return;
        }
      }
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    } else {
      // comm in row parallel & last col parallel & grad comm in row parallel
      if (ds.get_dim(-1) >= 2 && ds.get_dim(0) >= 2) {
        auto new_ds = dup2split(ds, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      // grad comm in last col parallel
      if (ds.get_dim(0) >= 2 && ds.get_dim(1) >= 2) {
        auto new_ds = split2split(ds, 1, 0);
        (std::dynamic_pointer_cast<CommOpImpl>(op->_body))->set_dst_distributed_states(new_ds);
        comm_output->set_distributed_states(new_ds);
        return;
      } 
      HT_LOG_WARN << "op " << op << " ds states = " << ds.get_states();
      HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
    }
  }
  HT_LOG_INFO << "op " << op << " is not a variable/placeholder nor a comm";
  HT_RUNTIME_ERROR << "Unreachable, some assumptions are wrong, plz inform Lhy";
}

} // namespace graph
} // namespace hetu
