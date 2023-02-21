#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/ops/OnesLike.h"
#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/Communicate.h"

#include "hetu/impl/communication/comm_group.h"
using namespace hetu::impl::comm;

namespace hetu {
namespace autograd {

TensorList Gradients(const TensorList& ys, const TensorList& xs,
                     const TensorList& grad_ys) {
  for (const auto& y : ys) {
    HT_ASSERT(y.is_defined()) << "Passed an empty operation linker.";
    HT_ASSERT(y->is_tensor()) << "Cannot compute the gradient of "
                              << "operator " << y->producer() << ".";
  }
  auto local_device = GetLocalDevice();
  auto topo_order = TopoSort(ys);
  // HT_LOG_INFO << "topo: " << topo_order;
  // forward: deduce distributed states
  for (auto it = topo_order.begin(); it != topo_order.end(); ++it) {
    auto& node = *it;
    for (auto& input : node->inputs()) {
      HT_ASSERT(input->get_distributed_states().is_valid()) 
               << "distributed states for input tensor is not valid!";
    }
    if (node->output(0)->get_distributed_states().is_valid()) {
      HT_LOG_DEBUG << local_device << ": " << node << ": output[0] states already valid: " << node->output(0)->get_distributed_states().print_states();
    } else {
      node->ForwardDeduceStates(); // inputs.ds -> outputs.ds
      // test: 这里假设每个op的output只有一个, 简单测试输出的ds, 如果op的output不只一个, 则只输出第0个的情况
      auto ds_grad = node->output(0)->get_distributed_states();
      // HT_LOG_DEBUG << node << ": states[-2] = " << ds_grad.get_dim(-2) << ", states[-1] = " << ds_grad.get_dim(-1) << ", states[0] = " << ds_grad.get_dim(0) << ", states[1] = " << ds_grad.get_dim(1);
      HT_LOG_DEBUG << local_device << ": " << node << ": output[0]: " << ds_grad.print_states();
    }
    // HT_LOG_INFO << "node " << node << " deduce states end!";
  }

  TensorList grads = _FillGrads(ys, grad_ys);

  Tensor2TensorListMap edge2grads;
  Tensor2TensorMap edge2reduce_grad;
  for (size_t i = 0; i < ys.size(); i++) {
    auto it = edge2grads.find(ys[i]->id());
    if (it == edge2grads.end())
      edge2grads[ys[i]->id()] = {grads[i]};
    else
      it->second.push_back(grads[i]);
    // states deduce
    auto grad_op = grads[i]->producer(); // oneslike_op
    grad_op->ForwardDeduceStates(); // loss->grad_loss
  }

  // ys = [loss], xs = [w1, w2, w3, ...], edge2grads={loss.id: [1,], }
  // traverse the forward graph in the reversed topo order
  // HT_LOG_INFO << "=======>backward begin!";
  // backward: 注: forward需要推出每个op的output tensor的states, 但backward可以直接用forward的信息
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    auto& node = *it;
    TensorList grad_outputs;
    if (node->num_outputs() > 0) {
      grad_outputs.reserve(node->num_outputs());
      for (auto& out_edge : node->outputs()) {
        auto grad = _Sum(edge2grads[out_edge->id()]);
        edge2reduce_grad[out_edge->id()] = grad;
        grad_outputs.push_back(grad);
      }
    }
    // HT_LOG_INFO << "node " << node << " output tensor grad get!";
    if (node->num_inputs() > 0) {
      // HT_LOG_INFO << "node " << node << " input tensor grads get begin!";
      auto grad_inputs = node->Gradient(grad_outputs);
      // HT_LOG_INFO << "node " << node << " input tensor grads get end!";

      // HT_LOG_INFO << "node " << node << " input tensor map to grads begin!";
      for (size_t i = 0; i < node->num_inputs(); i++) {
        if (!grad_inputs[i].is_defined())
          continue;
        // states deduce
        // TODO: 这里只考虑了单个output的情况，还要考虑多个output的情况
        auto grad_op = grad_inputs[i]->producer(); // special case: sigmoid_op
        // if (is_comm_op(grad_op)) {
        //   HT_LOG_INFO << "backward: comm_op: " << grad_op;
        // }
        grad_op->ForwardDeduceStates(); // 理论上在op definiiton的时候就已经推导完ds了, 这句话是多余的

        // 如果grad是partial的, 则将其转为duplicate
        auto ds_grad = grad_inputs[i]->get_distributed_states();

        // HT_LOG_DEBUG << "grad_op: " << grad_op << ", states[-2] = " << ds_grad.get_dim(-2) << ", states[-1] = " << ds_grad.get_dim(-1) << ", states[0] = " << ds_grad.get_dim(0) << ", states[1] = " << ds_grad.get_dim(1);
        HT_LOG_DEBUG << local_device << ": " << "grad_op: " << grad_op << ": " << ds_grad.print_states() << ", shape: " << grad_inputs[i]->shape();
        
        Tensor final_grad = grad_inputs[i];
        // 只需要考虑一个问题: 什么情况下, op左侧的操作数的梯度会是partial的?
        // 在矩阵乘法中: MatMul(a, b) = c, 则对于grad_a来说, a的duplicate维==b的第1维=grad_a的partial维
        // 对于grad_b来说, b的duplicate维==a的第1维==grad_b的partial维
        // 显然, 对每个device来说, 原先a/b是duplicate的, 所需的grad_a/grad_b也应该是duplicate的, 但反向计算得到的grad_a/grad_b却是partial的, 因此需要把partial通过reduce转化为duplicate
        if (ds_grad.get_dim(-2) > 1) { // partial->duplicate
          int32_t device_num = ds_grad.get_device_num();
          std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
          std::unordered_map<int32_t, int32_t> res_states = ds_grad.combine_states(src2dst);
          std::vector<int32_t> res_order = ds_grad.combine_order(src2dst);
          DistributedStates ds_dst({device_num, res_states, res_order});
          // HT_LOG_DEBUG << "backward: partial to duplicate: " << grad_inputs[i] << ", dst states: states[-2] = " << res_states[-2] 
          // << ", states[-1] = " << res_states[-1] << ", states[0] = " << res_states[0] << ", states[1] = " << res_states[1];
          HT_LOG_DEBUG << local_device << ": " << "backward: partial to duplicate: " << grad_inputs[i] << ", dst states = " << ds_dst.print_states();
          final_grad = CommOp(grad_inputs[i], ds_dst, OpMeta().set_name("comm_op_after_" + grad_op->name()))->output(0); // allreduce
          final_grad->producer()->ForwardDeduceStates(); // 别忘了任何一个新生成的op都要推导ds
        }

        // map: {edge: [..., grad_edge]}          
        auto in_edge = node->input(i);
        auto it = edge2grads.find(in_edge->id());
        if (it == edge2grads.end())
          edge2grads[in_edge->id()] = {final_grad};
        else
          it->second.push_back(final_grad);
      }
      // HT_LOG_INFO << "node " << node << " input tensor map to grads end!";
    }
  }
  // HT_LOG_DEBUG << "=======>backward end!";
  TensorList ret;
  ret.reserve(xs.size());
  for (auto& x : xs) {
    auto it = edge2reduce_grad.find(x->id());
    if (it != edge2reduce_grad.end())
      ret.push_back(it->second);
    else
      ret.emplace_back();
  }
  return ret;
}

TensorList _FillGrads(const TensorList& edges, const TensorList& grads) {
  TensorList ret;
  ret.reserve(edges.size());
  if (grads.empty()) {
    // fill ones for scalar nodes
    for (const auto& edge : edges) {
      // TODO(ffc): check whether require grad?
      ret.emplace_back(OnesLikeOp(edge)->output(0));
    }
  } else {
    HT_ASSERT_EQ(edges.size(), grads.size())
      << "Provided " << edges.size() << " variables and " << grads.size()
      << " grads.";
    for (size_t i = 0; i < edges.size(); i++) {
      const auto& edge = edges[i];
      const auto& grad = grads[i];
      if (!grad.is_defined()) {
        ret.emplace_back(OnesLikeOp(edge)->output(0));
      } else {
        ret.emplace_back(grad);
      }
    }
  }
  return ret;
}

TensorList _Filter(const TensorList& edges) {
  TensorList filtered;
  filtered.reserve(edges.size());
  std::copy_if(edges.begin(), edges.end(), std::back_inserter(filtered),
               [](const Tensor& e) { return e.is_defined(); });
  return filtered;
}

Tensor _Sum(const TensorList& edges) {
  TensorList filtered = _Filter(edges);
  if (filtered.empty())
    return Tensor();
  else if (edges.size() == 1)
    return edges[0];
  else {
    auto sum_op = SumOp(edges);
    sum_op->ForwardDeduceStates(); // default copy
    return sum_op->output(0);
  }
}

} // namespace autograd
} // namespace hetu
