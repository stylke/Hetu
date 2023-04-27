#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class TransposeOpImpl;
class TransposeOp;

class TransposeOpImpl : public OpInterface {
 public:
  TransposeOpImpl(HTShape perms)
  : OpInterface(quote(TransposeOp)), _perms(perms) {
  }

  HTShape get_perms() const {
    return _perms;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape res_shape = {};
    if (inputs[0]->has_shape()) {
      HTShape ori_shape = inputs[0]->shape();
      HTShape perm = _perms;
      HT_ASSERT(perm.size() == ori_shape.size())
        << "Invalid perm size:" << _perms << ",expect:" << inputs[0]->shape();
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
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                .set_shape(res_shape)
                                .set_device(inputs[0]->device());
    return {output_meta};       
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _perms;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const TransposeOpImpl&>(rhs);
      return (get_perms() == rhs_.get_perms());
    }
    return false;
  }
};

Tensor MakeTransposeOp(Tensor input, HTShape perms, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
