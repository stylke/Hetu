#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SinOpImpl;
class SinOp;
class CosOpImpl;
class CosOp;

class SinOpImpl : public OpInterface {
 private:
  friend class SinOp;
  struct constrcutor_access_key {};

 public:
  SinOpImpl(bool inplace)
  : OpInterface(quote(SinOp)), _inplace(inplace) {
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
      const auto& rhs_ = reinterpret_cast<const SinOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeSinOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeSinInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class SinGradientOpImpl : public OpInterface {

 public:
  SinGradientOpImpl()
  : OpInterface(quote(SinGradientOp)) {
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

Tensor MakeSinGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

class CosOpImpl : public OpInterface {

 public:
  CosOpImpl(bool inplace)
  : OpInterface(quote(CosOp)), _inplace(inplace) {
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
      const auto& rhs_ = reinterpret_cast<const CosOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeCosOp(Tensor input, bool inplace = false, OpMeta op_meta = OpMeta());

class CosGradientOpImpl : public OpInterface {

 public:
  CosGradientOpImpl()
  : OpInterface(quote(CosGradientOp)) {
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

Tensor MakeCosGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
