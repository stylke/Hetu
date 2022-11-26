#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class TransposeOpDef;
class TransposeOp;

class TransposeOpDef : public OperatorDef {
 private:
  friend class TransposeOp;
  struct constrcutor_access_key {};

 public:
  TransposeOpDef(const constrcutor_access_key&, Tensor input, HTShape perms,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(TransposeOp), {input}, op_meta), _perms(perms) {
    HTShape res_shape = {};
    if (input->has_shape()) {
      HTShape ori_shape = input->shape();
      HTShape perm = perms;
      HT_ASSERT(perm.size() == ori_shape.size())
        << "Invalid perm size:" << perms << ",expect:" << input->shape();
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
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(res_shape));
  }

  HTShape get_perms() const {
    return _perms;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _perms;
};

class TransposeOp final : public OpWrapper<TransposeOpDef> {
 public:
  TransposeOp(Tensor input, HTShape perms, const OpMeta& op_meta = OpMeta())
  : OpWrapper<TransposeOpDef>(make_ptr<TransposeOpDef>(
      TransposeOpDef::constrcutor_access_key(), input, perms, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
