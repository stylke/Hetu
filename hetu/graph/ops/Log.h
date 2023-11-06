#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LogOpImpl;
class LogOp;
class LogGradientOpImpl;
class LogGradientOp;

class LogOpImpl : public OpInterface {
 private:
  friend class LogOp;
  struct constrcutor_access_key {};

 public:
  LogOpImpl(bool inplace)
  : OpInterface(quote(LogOp)), _inplace(inplace) {
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
      const auto& rhs_ = reinterpret_cast<const LogOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeLogOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeLogInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
