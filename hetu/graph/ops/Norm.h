#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class NormOpImpl;
class NormOp;
class NormGradientOpImpl;
class NormGradientOp;

class NormOpImpl : public OpInterface {
 public:
  NormOpImpl(int64_t p = 1, int64_t dim = 0, bool keepdim = false)
  : OpInterface(quote(NormOp)),
  _p(p),
  _dim(dim),
  _keepdim(keepdim) {
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
    return _dim;
  }

  inline bool keepdim() const{
    return _keepdim;
  }
  
 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape outshape = inputs[0]->shape();
    int64_t axi = dim() >= 0? dim(): dim() + outshape.size();
    if (keepdim()) 
      outshape[axi] = 1;
    else 
      outshape.erase(outshape.begin() + axi);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(outshape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _p;

  int64_t _dim;

  bool _keepdim;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const NormOpImpl&>(rhs);
      return (getp() == rhs_.getp()
              && dim() == rhs_.dim()
              && keepdim() == rhs_.keepdim());
    }
    return false;
  }
};

Tensor MakeNormOp(Tensor input, int64_t p = 1, int64_t dim = 0, 
                  bool keepdim = false, const OpMeta& op_meta = OpMeta());

class NormGradientOpImpl : public OpInterface {
 public:
  NormGradientOpImpl(int64_t p, int64_t dim)
  : OpInterface(quote(NormGradientOp)),
  _p(p),
  _dim(dim) {
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
    return _dim;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _p;

  int64_t _dim;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const NormGradientOpImpl&>(rhs);
      return (getp() == rhs_.getp()
              && dim() == rhs_.dim());
    }
    return false;
  }
};

Tensor MakeNormGradientOp(Tensor input, Tensor output, Tensor grad_output, int64_t p, 
                          int64_t dim, const OpMeta& op_meta = OpMeta());

} // namespace graph
} // namespace hetu
