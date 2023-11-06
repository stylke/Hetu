#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ExpOpImpl;
class ExpOp;
class ExpGradientOpImpl;
class ExpGradientOp;

class ExpOpImpl : public OpInterface {
 private:
  friend class ExpOp;
  struct constrcutor_access_key {};

 public:
  ExpOpImpl(bool inplace)
  : OpInterface(quote(ExpOp)), _inplace(inplace) {
  }

  inline bool inplace() const{
    return _inplace;
  }


 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ExpOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeExpOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeExpInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
