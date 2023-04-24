#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/BroadcastShape.h"
#include "hetu/autograd/ops/ReduceMean.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ReduceMeanOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::ReduceMean, inputs.at(0),
    outputs.at(0), get_axes().data(), get_axes().size(), stream());
}

TensorList ReduceMeanOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad =
    BroadcastShapeOp(grad_outputs.at(0), HTShape(), HTAxes(), g_op_meta)
      ->output(0);
  grad = MulByConstOp(grad, 0, g_op_meta.set_name(grad_name()))->output(0);
  return {grad};
}

HTShapeList ReduceMeanOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape input_shape = input_shapes.at(0);
  int ndim = input_shape.size();
  int64_t mean_multiplier = 1;
  HTShape axes = get_axes();
  int len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  set_grad_shape(input_shape);
  // BroadcastShapeOp& grad_b = reinterpret_cast<BroadcastShapeOp&>(grad);
  // MulByConstOp grad_a = *(MulByConstOp*)&grad_;
  // if (grad_b)
  //   grad_b->set_shape(input_shape);
  HTShape add_axes = {};
  for (int i = 0; i < len; ++i) {
    if (axes[i] < 0) {
      axes[i] += ndim;
    }
    HT_ASSERT(axes[i] >= 0 && axes[i] < ndim);
    mean_multiplier *= input_shape[axes[i]];
    if (keepdims[i] == true)
      input_shape[axes[i]] = 1;
    else {
      input_shape[axes[i]] = 0;
      add_axes.emplace_back(axes[i]);
    }
  }
  set_grad_axes(add_axes);
  set_grad_const(1.0 / mean_multiplier);
  // if (grad_b)
  //   grad_b->set_add_axes(add_axes);
  // if (grad_a)
  //   grad_a->set_const_value(1.0/mean_multiplier);
  HTShape output_shape(0);
  for (int i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  return {output_shape};
}

void ReduceMeanOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ReduceMeanOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Tensor input shouldn't be partial!";
  HTShape axes = get_axes();
  HTKeepDims keepdims = get_keepdims();  
  int32_t partial = ds_input.get_dim(-2);
  // device_num
  int32_t device_num = ds_input.get_device_num();
  // states
  std::unordered_map<int32_t, int32_t> states = ds_input.get_states();
  for (auto d : axes) {
    int32_t state_d = ds_input.get_dim(d); 
    if (state_d > 1) {
      partial *= state_d;
      states.erase(d);
    }
  }
  states[-2] = partial;
  std::vector<int32_t> sorted_keys;
  for (auto& pair : states) {
    if (pair.first >= 0) {
      sorted_keys.push_back(pair.first);
    }
  }
  std::sort(sorted_keys.begin(), sorted_keys.end());
  for (auto d : sorted_keys) {
    int32_t reduce_dimensions = 0;
    for (int i = 0; i < axes.size(); i++) {
      if (axes[i] < d && !keepdims[i]) {
        reduce_dimensions++;
      }
    }
    int32_t new_d = d - reduce_dimensions;
    if (new_d != d) {
      states[new_d] = states[d];
      states.erase(d);
    }
  }
  // order
  std::vector<int32_t> order = ds_input.get_order();
  int32_t dup_occur = 0;
  bool prev_dup = false;
  std::vector<int64_t> partial_candidate = axes;
  partial_candidate.push_back(-2);
  for (int i = order.size() - 1; i >= 0; i--) {
    if (std::find(partial_candidate.begin(), partial_candidate.end(), order[i]) != partial_candidate.end()) {
      if (!prev_dup) {
        dup_occur++;
      }
      prev_dup = true;
      if (order[i] != -2) {
        if (std::find(order.begin(), order.end(), -2) == order.end()) {
          order[i] = -2;
        } else {
          order.erase(order.begin() + i);
        }
      }
    } else {
      prev_dup = false;
    }
  }
  HT_ASSERT(dup_occur <= 1) << "Duplicate dimension and reduce dimensions must be consecutive!";
  for (int i = 0;i < order.size(); i++) {
    int32_t reduce_dimensions = 0;
    for (int j = 0; j < axes.size(); j++) {
      if (axes[j] < order[i] && !keepdims[j]) {
        reduce_dimensions++;
      }
    }
    order[i] -= reduce_dimensions;
  }
  std::ostringstream out;
  for (auto state : states) {
    out << "states[" << state.first << "] = " << state.second << "; ";
  }
  out << "\norder: ";
  for (auto o : order) {
    out << o << " ";
  }
  HT_LOG_INFO << out.str();
  _outputs[0]->set_distributed_states({device_num, states, order});
}

} // namespace autograd
} // namespace hetu
