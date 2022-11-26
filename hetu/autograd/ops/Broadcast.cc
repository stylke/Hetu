#include "hetu/autograd/ops/Broadcast.h"
#include "hetu/autograd/ops/Reduce.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BroadcastOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  if (mode() == 0) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::Broadcast, inputs.at(0),
                                    outputs.at(0), stream());
  } else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), get_add_axes(), stream());
  }
}

TensorList BroadcastOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = BroadcastGradientOp(grad_outputs.at(0), _outputs[0],
                                        get_grad_axes(), get_grad_keep_dims(),
                                        grad_op_meta().set_name(grad_name()))
                      ->output(0);
  if (mode() == 0)
    return {grad_input, Tensor()};
  else
    return {grad_input};
}

HTShapeList BroadcastOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  if (mode() == 0) {
    HTShape input_shape = input_shapes.at(0);
    HTShape output_shape = input_shapes.at(1);
    size_t ndim = output_shape.size();
    HT_ASSERT(input_shape.size() <= ndim);
    size_t diff = ndim - input_shape.size();
    HTAxes add_axes(diff);
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
      HT_ASSERT(output_shape[i] > 0);
      HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
      if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
        add_axes.emplace_back(i);
        keep_dims.emplace_back(true);
      }
    }
    set_grad_axes(add_axes);
    set_grad_keep_dims(keep_dims);
    outputlist = {input_shapes.at(1)};
  } else {
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
    }
    outputlist = {output_shape};
  }
  return outputlist;
}

void BroadcastGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::ReduceSum, inputs.at(0),
    outputs.at(0), get_axes().data(), get_axes().size(), stream());
}

HTShapeList
BroadcastGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  BroadcastOp& input_ptr =
    reinterpret_cast<BroadcastOp&>(_inputs[1]->producer());
  if (input_ptr) {
    set_axes(input_ptr->get_grad_axes());
    set_keepdims(input_ptr->get_grad_keep_dims());
  }
  HTShape input_shape = input_shapes.at(0);
  int ndim = input_shape.size();
  HTShape axes = get_axes();
  int len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  set_grad_shape(input_shape);
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
  HTShape output_shape(0);
  for (int i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  outputlist = {output_shape};
  return outputlist;
}

} // namespace autograd
} // namespace hetu
