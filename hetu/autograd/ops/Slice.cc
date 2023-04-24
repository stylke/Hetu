#include "hetu/autograd/ops/Slice.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SliceOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Slice,
                                  inputs.at(0), outputs.at(0),
                                  get_begin_pos().data(), stream());
}

TensorList SliceOpDef::DoGradient(const TensorList& grad_outputs) {
  return {SliceGradientOp(grad_outputs.at(0), _outputs[0], get_begin_pos(),
                          get_ori_output_shape(),
                          grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList SliceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape ori_shape = input_shapes.at(0);
  HT_ASSERT(ori_shape.size() == get_begin_pos().size());
  int ndim = ori_shape.size();
  HTShape ori_output_shape = get_ori_output_shape();
  HTShape output_shape = get_output_shape();
  HTShape begin_pos = get_begin_pos();
  for (int i = 0; i < ndim; ++i) {
    if (ori_output_shape[i] == -1) {
      output_shape[i] = ori_shape[i] - begin_pos[i];
    }
    HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(begin_pos[i] + output_shape[i] <= ori_shape[i]);
  }
  set_ori_output_shape(ori_shape);
  set_output_shape(output_shape);
  set_grad_output_shape(ori_shape);
  return {output_shape};
}

void SliceOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "SliceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  // HT_ASSERT(ds_input.check_pure_duplicate())
  //   << "Input tensor cannot be splited in any dimension!";
  HTShape ori_shape = _inputs[0]->shape();
  int ndim = ori_shape.size();
  HTShape ori_output_shape = get_ori_output_shape();
  HTShape begin_pos = get_begin_pos();
  for (int i = 0; i < ndim; i++) {
    if (!(begin_pos[i] == 0 && (ori_output_shape[i] == -1 || begin_pos[i] + ori_output_shape[i] == ori_shape[i]))) {
      HT_ASSERT(ds_input.get_dim(i) == 1)
        << "Slice dimension " << i << " shouldn't be splited!"; 
    }
  }
  _outputs[0]->set_distributed_states(ds_input);      
}

void SliceGradientOpDef::DoCompute(const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::SliceGradient, inputs.at(0),
    outputs.at(0), get_begin_pos().data(), stream());
}

HTShapeList SliceGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  SliceOp& input_ptr = reinterpret_cast<SliceOp&>(_inputs[1]->producer());
  if (input_ptr) {
    set_output_shape(input_ptr->get_grad_output_shape());
  }
  HTShape output_shape = get_output_shape();
  HTShape begin_pos = get_begin_pos();
  HT_ASSERT(output_shape.size() > 0);
  HTShape ori_shape = input_shapes.at(0);
  HT_ASSERT(ori_shape.size() == begin_pos.size());
  int ndim = ori_shape.size();
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] + ori_shape[i] <= output_shape[i]);
  }
  set_ori_output_shape(ori_shape);
  return {output_shape};
}

void SliceGradientOpDef::DeduceStates() {
  DistributedStates ds_grad_output = _inputs[0]->get_distributed_states();  
  DistributedStates ds_ori_input = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_grad_output.is_valid() && ds_ori_input.is_valid()
            && ds_grad_output.get_device_num() == ds_ori_input.get_device_num())
    << "SliceGradientOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_grad_output.get_dim(-2) == 1 && ds_ori_input.get_dim(-2) == 1)
    << "Tensor inputs shouldn't be partial!";
  HT_ASSERT(ds_grad_output.check_equal(ds_ori_input))
    << "Distributed states among grad_output and ori_input should be equal!";

  _outputs[0]->set_distributed_states(ds_grad_output);  
}

} // namespace autograd
} // namespace hetu
