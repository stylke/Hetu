#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/ops/OnesLike.h"
#include "hetu/autograd/ops/Sum.h"

namespace hetu {
namespace autograd {

TensorList Gradients(const TensorList& ys, const TensorList& xs,
                     const TensorList& grad_ys) {
  for (const auto& y : ys) {
    HT_ASSERT(y.is_defined()) << "Passed an empty operation linker.";
    HT_ASSERT(y->is_tensor()) << "Cannot compute the gradient of "
                              << "operator " << y->producer() << ".";
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
  }

  // traverse the forward graph in the reversed topo order
  auto topo_order = TopoSort(ys);
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

    if (node->num_inputs() > 0) {
      auto grad_inputs = node->Gradient(grad_outputs);
      for (size_t i = 0; i < node->num_inputs(); i++) {
        if (!grad_inputs[i].is_defined())
          continue;
        auto in_edge = node->input(i);
        auto it = edge2grads.find(in_edge->id());
        if (it == edge2grads.end())
          edge2grads[in_edge->id()] = {grad_inputs[i]};
        else
          it->second.push_back(grad_inputs[i]);
        // HT_LOG_INFO << node->name() << " " << node->input(i)->name() << " " << node->input(i)->shape();
      }
    }
  }

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
  else
    return SumOp(edges)->output(0);
}

} // namespace autograd
} // namespace hetu
