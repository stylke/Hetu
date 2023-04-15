#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class MaskedfillOpImpl;
class MaskedfillOp;

class MaskedfillOpImpl : public OpInterface {
 private:
  friend class MaskedfillOp;
  struct constrcutor_access_key {};

 public:
  MaskedfillOpImpl(double val)
  : OpInterface(quote(MaskedfillOp)),
  _val(val) {
  }

  inline double val() const {
    return _val;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _val;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MaskedfillOpImpl&>(rhs);
      return (val() == rhs_.val()); 
    }
    return false;
  }
};

Tensor MakeMaskedfillOp(Tensor input, Tensor mask, double val,
                        OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
