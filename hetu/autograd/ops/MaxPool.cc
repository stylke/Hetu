#include "hetu/autograd/ops/MaxPool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void MaxPoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::MaxPool,
                               inputs.at(0), get_kernel_H(), get_kernel_W(),
                               outputs.at(0), get_padding(), get_stride(),
                               stream());
}

TensorList MaxPoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {MaxPoolGradientOp(_outputs[0], grad_outputs.at(0), _inputs[0],
                            get_kernel_H(), get_kernel_W(), get_padding(),
                            get_stride(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList MaxPoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void MaxPoolOpDef::DeduceStates() {
  DistributedStates ds = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "MaxPoolOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds.get_dim(2) == 1 && ds.get_dim(3) == 1)
    << "H & W dimension shouldn't be splited, H: "
    << ds.get_dim(2) << ", W: " << ds.get_dim(3);
  _outputs[0]->set_distributed_states(ds);
}

void MaxPoolGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::MaxPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), stream());
}

HTShapeList
MaxPoolGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void MaxPoolGradientOpDef::DeduceStates() {
  DistributedStates ds_output = _inputs[0]->get_distributed_states();
  DistributedStates ds_output_grad = _inputs[1]->get_distributed_states();
  DistributedStates ds_input = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_output.is_valid() && ds_output_grad.is_valid() && ds_input.is_valid()
            && ds_output.get_device_num() == ds_output_grad.get_device_num() 
            && ds_output_grad.get_device_num() == ds_input.get_device_num())
    << "MaxPoolGradientOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_output.get_dim(-2) == 1 && ds_output_grad.get_dim(-2) == 1 && ds_input.get_dim(-2) == 1)
    << "Inputs tensor shouldn't be partial!";
  HT_ASSERT(ds_output_grad.check_equal(ds_output) && ds_output.check_equal(ds_input))
    << "Distributed states among output_grad, output and input should be equal!";
  HT_ASSERT(ds_input.get_dim(2) == 1 && ds_input.get_dim(3) == 1)
    << "H & W dimension of output_grad & output & input shouldn't be splited!";
  _outputs[0]->set_distributed_states(ds_input);
}

} // namespace autograd
} // namespace hetu
