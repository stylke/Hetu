#include "hetu/autograd/ops/Conv2dBroadcast.h"
#include "hetu/autograd/ops/Conv2dReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void Conv2dBroadcastOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Conv2dBroadcast, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList Conv2dBroadcastOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    Conv2dReduceSumOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList
Conv2dBroadcastOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(1)};
}
} // namespace autograd
} // namespace hetu
