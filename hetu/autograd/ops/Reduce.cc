#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/Broadcast.h"
#include "hetu/autograd/ops/Reduce.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ReduceOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  if (reduction() == ReductionType::MEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::ReduceMean, inputs.at(0),
      outputs.at(0), get_axes().data(), get_axes().size(), stream());
    // HT_LOG_INFO << inputs.at(0) << "\n" << outputs.at(0) << "\n" << get_axes();
  } else if (reduction() == ReductionType::SUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::ReduceSum, inputs.at(0),
      outputs.at(0), get_axes().data(), get_axes().size(), stream());
  }
}

TensorList ReduceOpDef::DoGradient(const TensorList& grad_outputs) {
  return {ReduceGradientOp(grad_outputs.at(0), _outputs[0], _inputs[0], HTShape(), reduction(),
                           get_axes(), get_keepdims(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void ReduceOpDef::DoInferMeta() {
  if (_axes.size() == 0) {
    _axes.reserve(_inputs[0]->ndim());
    for (size_t i = 0; i < _inputs[0]->ndim(); ++i) {
      _axes.push_back(i);
    }
  }
  _axes = NDArrayMeta::ParseAxes(_axes, _inputs[0]->ndim());
  // HT_LOG_INFO << axes << " " << NDArrayMeta::ParseAxes(axes, input->ndim());
  HT_ASSERT(_keepdims.size() == _axes.size() || _keepdims.size() == 1);
  if (_keepdims.size() == 1) {
    int len = _axes.size();
    bool keepdim = _keepdims[0];
    for (int i = 1; i < len; ++i) {
      _keepdims.emplace_back(keepdim);
    }
  }
  HTShape output_shape;
  if (_inputs[0]->has_shape()) {
    int ndim = _inputs[0]->ndim();
    HTShape tmp_axes = _axes;
    HTShape input_shape = _inputs[0]->shape();
    int len = tmp_axes.size();
    for (int i = 0; i < len; ++i) {
      if (tmp_axes[i] < 0) {
        tmp_axes[i] += ndim;
      }
      HT_ASSERT(tmp_axes[i] >= 0 && tmp_axes[i] < ndim)
        << "axes:" << tmp_axes[i] << " ,ndims:" << ndim;
      if (_keepdims[i] == true)
        input_shape[tmp_axes[i]] = 1;
      else
        input_shape[tmp_axes[i]] = 0;
    }
    for (int i = 0; i < ndim; ++i) {
      if (input_shape[i] > 0)
        output_shape.emplace_back(input_shape[i]);
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(1);
  }
  // HT_LOG_INFO << _axes << " " << _keepdims << " " << input->shape() << " " << output_shape;
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape).set_device(_inputs[0]->device()));
}

HTShapeList ReduceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  // HT_LOG_INFO << input_shapes;
  if (reduction() == ReductionType::MEAN) {
    HTShape input_shape = input_shapes.at(0);
    int ndim = input_shape.size();
    int64_t mean_multiplier = 1;
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
    HTShape output_shape(0);
    for (int i = 0; i < ndim; ++i) {
      if (input_shape[i] > 0)
        output_shape.emplace_back(input_shape[i]);
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(1);
    outputlist = {output_shape};
  } else if (reduction() == ReductionType::SUM) {
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
  }
  // HT_LOG_INFO << outputlist;
  return outputlist;
}

void ReduceGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  if (reduction() == ReductionType::MEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(0),
      get_const_value(), outputs.at(0), get_add_axes(), stream());
  } else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), get_add_axes(), stream());
  }
}

void ReduceGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[2]->meta());
}

HTShapeList ReduceGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  if (reduction() == ReductionType::MEAN) {
    HTShape input_shape = input_shapes.at(2);
    int ndim = input_shape.size();
    int64_t mean_multiplier = 1;
    HTShape axes = get_axes();
    int len = axes.size();
    HTKeepDims keepdims = get_keepdims();
    set_shape(input_shape);
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
    set_add_axes(add_axes);
    set_const_value(1.0 / mean_multiplier);
  } else if (reduction() == ReductionType::SUM) {
    HTShape input_shape = input_shapes.at(2);
    int ndim = input_shape.size();
    HTShape axes = get_axes();
    int len = axes.size();
    HTKeepDims keepdims = get_keepdims();
    set_shape(input_shape);
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
    set_add_axes(add_axes);
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
    // set_grad_axes(add_axes);
    // set_grad_keep_dims(keep_dims);
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
  }
  outputlist = {output_shape};
  return outputlist;
}

} // namespace autograd
} // namespace hetu
