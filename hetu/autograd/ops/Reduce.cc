#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/Broadcast.h"
#include "hetu/autograd/ops/Reduce.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ReduceOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  if (mode() == "mean") {
    HT_DISPATCH_KERNEL_CUDA_ONLY(
      placement().type(), type(), hetu::impl::ReduceMean, inputs.at(0),
      outputs.at(0), get_axes().data(), get_axes().size(), stream());
  } else if (mode() == "sum") {
    HT_DISPATCH_KERNEL_CUDA_ONLY(
      placement().type(), type(), hetu::impl::ReduceSum, inputs.at(0),
      outputs.at(0), get_axes().data(), get_axes().size(), stream());
  }
}

TensorList ReduceOpDef::DoGradient(const TensorList& grad_outputs) {
  return {ReduceGradientOp(grad_outputs.at(0), _outputs[0], HTShape(), mode(),
                           HTAxes(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList ReduceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  if (mode() == "mean") {
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
    outputlist = {output_shape};
  } else if (mode() == "sum") {
    HTShape input_shape = input_shapes.at(0);
    int ndim = input_shape.size();
    HTShape axes = get_axes();
    int len = axes.size();
    HTKeepDims keepdims = get_keepdims();
    set_grad_shape(input_shape);
    // BroadcastShapeOp& grad_b = reinterpret_cast<BroadcastShapeOp&>(grad);
    // if (grad_b)
    //   grad_b->set_shape(input_shape);
    HTShape add_axes = {};
    for (int i = 0; i < len; ++i) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
      HT_ASSERT(axes[i] >= 0 && axes[i] < ndim);
      if (keepdims[i] == true)
        input_shape[axes[i]] = 1;
      else {
        input_shape[axes[i]] = 0;
        add_axes.emplace_back(axes[i]);
      }
    }
    set_grad_axes(add_axes);
    // if (grad_b)
    //   grad_b->set_add_axes(add_axes);
    HTShape output_shape(0);
    for (int i = 0; i < ndim; ++i) {
      if (input_shape[i] > 0)
        output_shape.emplace_back(input_shape[i]);
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(1);
    outputlist = {output_shape};
  }
  return outputlist;
}

void ReduceOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ReduceOpDef: distributed states for input must be valid!";
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
  _outputs[0]->set_distributed_states({device_num, states, order});
}

void ReduceGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  if (mode() == "mean") {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(0),
      get_const_value(), outputs.at(0), get_add_axes(), stream());
  } else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), get_add_axes(), stream());
  }
}

HTShapeList ReduceGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  ReduceOp& input_ptr = reinterpret_cast<ReduceOp&>(_inputs[1]->producer());
  if (input_ptr) {
    set_add_axes(input_ptr->get_grad_axes());
    set_shape(input_ptr->get_grad_shape());
    if (mode() == "mean") {
      set_const_value(input_ptr->get_grad_const());
    }
  }
  HTShape input_shape = input_shapes.at(0);
  HTShape output_shape = get_shape();
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim)
    << "input dim > output dim, input size is " << input_shape
    << " and output size is " << output_shape;
  size_t diff = ndim - input_shape.size();
  HTShape add_axes = get_add_axes();
  if (add_axes.size() > 0) {
    size_t asize = add_axes.size();
    HT_ASSERT(diff == asize ||
              (input_shape.size() == 1 && input_shape[0] == 1));
    HTKeepDims keep_dims(asize);
    for (size_t i = 0; i < asize; ++i) {
      keep_dims[i] = false;
      HT_ASSERT((size_t) add_axes[i] < ndim);
    }
    int64_t in_ind = 0;
    for (size_t i = 0; i < ndim; ++i) {
      bool flag = false;
      for (size_t j = 0; j < asize; ++j) {
        if (i == (size_t) add_axes[j]) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        HT_ASSERT(input_shape[in_ind] == output_shape[i])
          << "input shape:" << input_shape << ",output shape:" << output_shape
          << ",add_axes:" << add_axes;
        in_ind++;
      }
    }
    set_grad_axes(add_axes);
    set_grad_keep_dims(keep_dims);
  } else {
    add_axes.resize(diff);
    HTKeepDims keep_dims(diff);
    size_t len = diff + input_shape.size();
    HTShape n_input_shape(len);
    for (size_t i = 0; i < diff; ++i) {
      add_axes[i] = i;
      keep_dims[i] = false;
      n_input_shape[i] = 1;
    }
    for (size_t i = diff; i < len; ++i) {
      n_input_shape[i] = input_shape[i - diff];
    }
    for (size_t i = 0; i < ndim; ++i) {
      if (output_shape[i] == -1) {
        output_shape[i] = n_input_shape[i];
      }
      HT_ASSERT(output_shape[i] > 0) << "has Invalid shape.";
      HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i])
        << "input shape can't broadcast to output shape.";
      if (i >= diff && n_input_shape[i] == 1 && output_shape[i] > 1) {
        add_axes.emplace_back(i);
        keep_dims.emplace_back(true);
      }
    }
    set_grad_axes(add_axes);
    set_grad_keep_dims(keep_dims);
    // ReduceSumOp& input_ptr = reinterpret_cast<ReduceSumOp&>(grad_input);
    // if (input_ptr) {
    //   input_ptr->set_axes(add_axes);
    //   input_ptr->set_keepdims(keep_dims);
    // }
  }
  outputlist = {output_shape};
  return outputlist;
}

void ReduceGradientOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_ori_input = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_ori_input.is_valid()
            && ds_input.get_device_num() == ds_ori_input.get_device_num())
    << "ReduceGradientOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_ori_input.get_dim(-2) == 1)
    << "Inputs tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_equal(ds_ori_input))
    << "Distributed states among input and ori_input should be equal!";
  HT_ASSERT(ds_input.check_max_dim(1)) // same as broadcast shape
    << "Only support data parallel!";
  _outputs[0]->set_distributed_states(ds_input);
}

} // namespace autograd
} // namespace hetu
