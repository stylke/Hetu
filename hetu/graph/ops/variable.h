#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/init/initializer.h"

namespace hetu {
namespace graph {

namespace {
inline DataType _InferDataType(const NDArray& data, DataType dtype) {
  return dtype != kUndeterminedDataType
    ? dtype
    : (data.is_defined() ? data->dtype() : kUndeterminedDataType);
}
} // namespace

class VariableOpImpl : public OpInterface {
 protected:
  VariableOpImpl(OpType&& type, const Initializer& init, HTShape shape,
                 DataType dtype, bool requires_grad)
  : OpInterface(std::move(type)),
    _init(init.copy()),
    _shape(std::move(shape)),
    _dtype(dtype),
    _requires_grad(requires_grad) {
    _check_init();
  }

  VariableOpImpl(OpType&& type, NDArray provided_data, bool copy_provided_data,
                 DataType dtype, bool requires_grad)
  : OpInterface(std::move(type)),
    _provided_data(std::move(provided_data)),
    _copy_provided_data(copy_provided_data),
    _shape(provided_data->shape()),
    _dtype(_InferDataType(provided_data, dtype)),
    _requires_grad(requires_grad) {
    _check_init();
  }

  void _check_init() {
    HT_VALUE_ERROR_IF(std::find(_shape.begin(), _shape.end(), -1) !=
                      _shape.end())
      << "Shape of " << _type << " is undetermined: " << _shape;
    HT_VALUE_ERROR_IF(_dtype == kUndeterminedDataType)
      << "Data type of " << _type << " is undetermined";
  }

 public:
  VariableOpImpl(const Initializer& init, HTShape shape,
                 DataType dtype = kFloat32, bool requires_grad = false)
  : VariableOpImpl(quote(VariableOp), init, std::move(shape), dtype,
                   requires_grad) {}

  VariableOpImpl(NDArray provided_data, bool copy_provided_data, DataType dtype,
                 bool requires_grad)
  : VariableOpImpl(quote(VariableOp), provided_data, copy_provided_data, dtype,
                   requires_grad) {}

  uint64_t op_indicator() const noexcept override {
    return VARIABLE_OP;
  }

 protected:
  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_id) const override;

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    return {NDArrayMeta().set_shape(shape()).set_dtype(dtype())};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    return {Tensor()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {shape()};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {}

 public:
  bool operator==(const OpInterface& rhs) const override {
    return false;
  }

  const Initializer& initializer() const {
    return *_init;
  }

  const HTShape& shape() const {
    return _shape;
  }

  DataType dtype() const {
    return _dtype;
  }

  bool requires_grad() const {
    return _requires_grad;
  }

 protected:
  std::shared_ptr<Initializer> _init;
  NDArray _provided_data;
  bool _copy_provided_data;
  HTShape _shape;
  DataType _dtype;
  bool _requires_grad;
};

Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype = kFloat32, bool requires_grad = false, 
                      const DistributedStates& ds = DistributedStates(), 
                      OpMeta op_meta = OpMeta());

Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data = false,
                      DataType dtype = kUndeterminedDataType, bool requires_grad = false, 
                      const DistributedStates& ds = DistributedStates(), 
                      OpMeta op_meta = OpMeta());

Tensor MakeParameterOp(const Initializer& init, HTShape shape,
                       DataType dtype = kFloat32, bool requires_grad = false, 
                       const DistributedStates& ds = DistributedStates(), 
                       OpMeta op_meta = OpMeta());

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data = false, 
                       DataType dtype = kUndeterminedDataType, bool requires_grad = false, 
                       const DistributedStates& ds = DistributedStates(),
                       OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
