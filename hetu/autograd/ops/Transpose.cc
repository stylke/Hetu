#include "hetu/autograd/ops/Transpose.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void TransposeOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Transpose, inputs.at(0),
                                  outputs.at(0), get_perms().data(), stream());
}

TensorList TransposeOpDef::DoGradient(const TensorList& grad_outputs) {
  const HTShape& perm = get_perms();
  HTShape grad_perm = perm;
  for (size_t i = 0; i < perm.size(); ++i) {
    grad_perm[perm[i]] = i;
  }
  return {TransposeOp(grad_outputs.at(0), grad_perm,
                      grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void TransposeOpDef::DoInferMeta() {
  HTShape res_shape = {};
  if (_inputs[0]->has_shape()) {
    HTShape ori_shape = _inputs[0]->shape();
    HTShape perm = _perms;
    HT_ASSERT(perm.size() == ori_shape.size())
      << "Invalid perm size:" << _perms << ",expect:" << _inputs[0]->shape();
    int ndim = perm.size();
    HTShape vis(ndim);
    for (int i = 0; i < ndim; ++i) {
      HT_ASSERT(perm[i] < ndim);
      HT_ASSERT(vis[perm[i]] == 0);
      vis[perm[i]]++;
    }
    res_shape = ori_shape;
    for (int i = 0; i < ndim; ++i) {
      res_shape[i] = ori_shape[perm[i]];
    }
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(res_shape).set_device(_inputs[0]->device()));
}

HTShapeList TransposeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape ori_shape = input_shapes.at(0);
  HTShape perm = get_perms();
  HT_ASSERT(perm.size() == ori_shape.size());
  int ndim = perm.size();
  HTShape vis(ndim);
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(perm[i] < ndim);
    HT_ASSERT(vis[perm[i]] == 0);
    vis[perm[i]]++;
  }
  HTShape res_shape = ori_shape;
  for (int i = 0; i < ndim; ++i) {
    res_shape[i] = ori_shape[perm[i]];
  }
  return {res_shape};
}

} // namespace autograd
} // namespace hetu
