#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SumOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  int len = inputs.size();
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, outputs.at(0), 0,
                                  stream());
  for (int i = 0; i < len; ++i) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::AddElewise, inputs.at(i),
                                    outputs.at(0), outputs.at(0), stream());
  }
}

TensorList SumOpDef::DoGradient(const TensorList& grad_outputs) {
  TensorList grad_inputs;
  grad_inputs.reserve(num_inputs());
  for (size_t i = 0; i < num_inputs(); i++)
    grad_inputs.push_back(grad_outputs.at(0));
  return grad_inputs;
}

HTShapeList SumOpDef::DoInferShape(const HTShapeList& input_shapes) {
  int len = input_shapes.size();
  size_t max_size = 0;
  int max_idx = 0;
  for (int idx = 0; idx < len; ++idx) {
    size_t tmp = 1;
    for (size_t i = 0; i < input_shapes.at(idx).size(); ++i) {
      tmp *= input_shapes.at(idx)[i];
    }
    if (tmp > max_size) {
      max_idx = idx;
      max_size = tmp;
    }
  }
  return {input_shapes.at(max_idx)};
}

} // namespace autograd
} // namespace hetu
