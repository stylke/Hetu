#include "hetu/autograd/ops/AvgPool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void AvgPoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::AvgPool,
                               inputs.at(0), get_kernel_H(), get_kernel_W(),
                               outputs.at(0), get_padding(), get_stride(),
                               stream());
}

TensorList AvgPoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {AvgPoolGradientOp(_outputs[0], grad_outputs.at(0), _inputs[0],
                            get_kernel_H(), get_kernel_W(), get_padding(),
                            get_stride(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList AvgPoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void AvgPoolOpDef::DeduceStates() {
  DistributedStates ds = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "AvgPoolOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds.get_dim(2) == 1 && ds.get_dim(3) == 1)
    << "H & W dimension shouldn't be splited, H: "
    << ds.get_dim(2) << ", W: " << ds.get_dim(3);
  _outputs[0]->set_distributed_states(ds);
}

void AvgPoolGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::AvgPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), stream());
}

HTShapeList
AvgPoolGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void AvgPoolGradientOpDef::DeduceStates() {
  DistributedStates ds_output = _inputs[0]->get_distributed_states();
  DistributedStates ds_output_grad = _inputs[1]->get_distributed_states();
  DistributedStates ds_input = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_output.is_valid() && ds_output_grad.is_valid() && ds_input.is_valid()
            && ds_output.get_device_num() == ds_output_grad.get_device_num() 
            && ds_output_grad.get_device_num() == ds_input.get_device_num())
    << "AvgPoolGradientOpDef: distributed states for inputs tensor must be valid!";
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
