#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class AbsOpImpl;
class AbsOp;
class AbsGradientOpImpl;
class AbsGradientOp;

class AbsOpImpl : public OpInterface {
 private:
  friend class AbsOp;
  struct constrcutor_access_key {};

 public:
  AbsOpImpl(bool inplace)
  : OpInterface(quote(AbsOp)), _inplace(inplace) {
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
      const auto& rhs_ = reinterpret_cast<const AbsOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeAbsOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeAbsInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class AbsGradientOpImpl : public OpInterface {

 public:
  AbsGradientOpImpl()
  : OpInterface(quote(AbsGradientOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs); 
  }
};

Tensor MakeAbsGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
