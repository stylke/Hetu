#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LayerNormOpImpl;
class LayerNormOp;
class LayerNormGradientOpImpl;
class LayerNormGradientOp;

class LayerNormOpImpl : public OpInterface {
 private:
  friend class LayerNormOp;
  struct constrcutor_access_key {};

 public:
  LayerNormOpImpl(const HTShape& normalized_shape, double eps = 0.01)
  : OpInterface(quote(LayerNormOp)),
  _normalized_shape(normalized_shape),
  _eps(eps) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape local_shape = inputs[0]->shape();
    int ndim = local_shape.size();
    local_shape[ndim - 1] = 1;
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(local_shape)
                                           .set_device(inputs[0]->device());
    return {inputs[0]->meta(), output_meta, output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LayerNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()
              && normalized_shape() == rhs_.normalized_shape()); 
    }
    return false;
  }
};

TensorList MakeLayerNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias, HTShape normalized_shape, 
                           double eps = 0.01, const OpMeta& op_meta = OpMeta());

class LayerNormGradientOpImpl : public OpInterface {
 public:
  LayerNormGradientOpImpl(HTShape normalized_shape, double eps)
  : OpInterface(quote(LayerNormGradientOp)),
  _normalized_shape(normalized_shape),
  _eps(eps) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta(), inputs[2]->meta(), inputs[2]->meta()};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LayerNormGradientOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeLayerNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                                   Tensor save_mean, Tensor save_var, HTShape normalized_shape,
                                   double eps, const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
