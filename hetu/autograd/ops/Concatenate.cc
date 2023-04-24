#include "hetu/autograd/ops/Concatenate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ConcatenateOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) {
  int num = num_inputs();
  size_t offset = 0;
  size_t axis = get_axis();
  for (int i = 0; i < num; ++i) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::Concatenate, inputs.at(i),
                                    outputs.at(0), axis, offset, stream());
    offset += inputs.at(i)->shape(axis);
  }
}

TensorList ConcatenateOpDef::DoGradient(const TensorList& grad_outputs) {
  grad_inputs = {};
  TensorList grads = {};
  int num = num_inputs();
  size_t axis = get_axis();
  auto g_op_meta = grad_op_meta();
  for (int i = 0; i < num; ++i) {
    auto grad_input = ConcatenateGradientOp(
      _inputs[i], grad_outputs.at(0), axis, g_op_meta.set_name(grad_name(i)));
    grad_inputs.emplace_back(grad_input);
    grads.emplace_back(grad_input->output(0));
  }
  return grads;
}

HTShapeList ConcatenateOpDef::DoInferShape(const HTShapeList& input_shapes) {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  int ind = 0;
  ConcatenateGradientOp tmpOp = grad_inputs.at(0);
  if (tmpOp)
    tmpOp->set_offset(0);
  ind += 1;
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        tmpOp = grad_inputs.at(i);
        if (tmpOp)
          tmpOp->set_offset(out_dim);
        ind += 1;
        out_dim += shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

void ConcatenateOpDef::DeduceStates() {
  for (auto input : _inputs) {
    DistributedStates ds_input = input->get_distributed_states();
    HT_ASSERT(ds_input.get_dim(get_axis()) == 1)
      << "Concat was not allowed in splited dimension: " << get_axis();
  }
  // 直接调用默认的states copy函数做检查和赋值
  OperatorDef::DeduceStates();
}

void ConcatenateGradientOpDef::DoCompute(const NDArrayList& inputs,
                                         NDArrayList& outputs,
                                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ConcatenateGradient, inputs.at(0),
    outputs.at(0), get_axis(), get_offset(), stream());
}

HTShapeList
ConcatenateGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void ConcatenateGradientOpDef::DeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_grad_output = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_grad_output.is_valid()) 
    << "ConcatenateGradientOpDef: distributed states for input and grad_output must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1) 
    << "Tensor input and grad_output shouldn't be partial";
  HT_ASSERT(ds_input.check_equal(ds_grad_output)) 
    << "Distributed states for tensor input and tensor gard_output must be equal!";
  HT_ASSERT(ds_input.get_dim(get_axis()) == 1)
    << "Concat was not allowed in splited dimension: " << get_axis();
  _outputs[0]->set_distributed_states(ds_input);
}

} // namespace autograd
} // namespace hetu
