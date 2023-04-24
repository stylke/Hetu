#include "hetu/autograd/ops/Split.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SplitOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Slice,
                                  inputs.at(0), outputs.at(0),
                                  get_begin_pos().data(), stream());
}

TensorList SplitOpDef::DoGradient(const TensorList& grad_outputs) {
  return {SplitGradientOp(grad_outputs.at(0), _outputs[0], get_axes(),
                          get_indices(), get_splits(), get_begin_pos(),
                          get_ori_output_shape(),
                          grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList SplitOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape ori_shape = input_shapes.at(0);
  int ndim = ori_shape.size();
  HTShape begin_pos(ndim);
  HTShape output_shape(ndim);
  for (int i = 0; i < ndim; ++i) {
    begin_pos[i] = 0;
    output_shape[i] = ori_shape[i];
  }
  HTShape axes = get_axes();
  HTShape indices = get_indices();
  HTShape splits = get_splits();
  int len = axes.size();
  for (int i = 0; i < len; ++i) {
    int64_t axe = axes[i];
    int64_t ind = indices[i];
    int64_t spl = splits[i];
    int64_t part_size = ori_shape[axe] / spl;
    begin_pos[axe] = ind * part_size;
    if (ind != spl - 1) {
      output_shape[axe] = part_size;
    } else {
      output_shape[axe] = ori_shape[axe] - begin_pos[axe];
    }
  }
  set_begin_pos(begin_pos);
  set_grad_begin_pos(begin_pos);
  set_grad_output_shape(ori_shape);
  set_output_shape(output_shape);
  set_ori_output_shape(ori_shape);
  return {output_shape};
}

void SplitOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid())
    << "SplitOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Tensor input shouldn't be partial!";
  HTShape axes = get_axes();    
  for (int i = 0; i < axes.size(); i++) {
    HT_ASSERT(ds_input.get_dim(axes[i]) == 1)
      << "Dimension in axes shouldn't be split: " << axes[i];
  }
  _outputs[0]->set_distributed_states(ds_input);    
}

void SplitGradientOpDef::DoCompute(const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::SliceGradient, inputs.at(0),
    outputs.at(0), get_begin_pos().data(), stream());
}

HTShapeList SplitGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  SplitOp& input_ptr = reinterpret_cast<SplitOp&>(_inputs[1]->producer());
  if (input_ptr) {
    set_begin_pos(input_ptr->get_grad_begin_pos());
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

void SplitGradientOpDef::DeduceStates() {
  DistributedStates ds_grad_output = _inputs[0]->get_distributed_states();  
  DistributedStates ds_ori_input = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_grad_output.is_valid() && ds_ori_input.is_valid()
            && ds_grad_output.get_device_num() == ds_ori_input.get_device_num())
    << "SliceGradientOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_grad_output.get_dim(-2) == 1 && ds_ori_input.get_dim(-2) == 1)
    << "Tensor inputs shouldn't be partial!";
  HT_ASSERT(ds_grad_output.check_equal(ds_ori_input))
    << "Distributed states among grad_output and ori_input should be equal!";
  HT_ASSERT(ds_grad_output.check_pure_duplicate())
    << "Tensor grad_output and ori_input shouldn't be splited!";
  _outputs[0]->set_distributed_states(ds_grad_output);  
}

} // namespace autograd
} // namespace hetu
