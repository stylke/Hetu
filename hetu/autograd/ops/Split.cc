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

} // namespace autograd
} // namespace hetu
