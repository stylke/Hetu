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

// class VariableOpImpl;
// class VariableOp;
// class ParallelVariableOpImpl;
// class ParallelVariableOp;

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

class ParallelVariableOpImpl : public OpInterface {
 public:
  ParallelVariableOpImpl(const Initializer& init, HTShape global_shape, 
                         const DistributedStates& ds, int64_t local_idx,
                         DataType dtype = kFloat32, bool requires_grad = false)
  : OpInterface(quote(ParallelVariableOp)), _init(init.copy()), 
    _global_shape(global_shape), _local_idx(local_idx), 
    _dtype(dtype), _ds(ds), _requires_grad(requires_grad) {
      _local_shape = get_local_shape(global_shape, ds);
    }

  HTShape get_local_shape(HTShape& global_shape, const DistributedStates& ds) {
    if (!_local_shape.empty())
      return _local_shape;
    HTShape shape(global_shape.size());
    for (size_t d = 0; d < global_shape.size(); d++) {
      shape[d] = global_shape[d] / ds.get_dim(d);
    }
    return shape;    
  }

  uint64_t op_indicator() const noexcept override {
    return VARIABLE_OP;
  }  

 protected:
  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_id) const override;

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    return {NDArrayMeta().set_shape(local_shape()).set_dtype(dtype())};
  }                     

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    return {Tensor()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {local_shape()};
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

  const HTShape& global_shape() const {
    return _global_shape;
  }

  const HTShape& local_shape() const {
    return _local_shape;
  }

  const DistributedStates& ds() const {
    return _ds;
  }

  int64_t local_idx() const {
    return _local_idx;
  }

  DataType dtype() const {
    return _dtype;
  }

  bool requires_grad() const {
    return _requires_grad;
  }

  std::shared_ptr<Initializer> _init;
  HTShape _global_shape;
  HTShape _local_shape;
  DistributedStates _ds;
  int64_t _local_idx;    
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

Tensor MakeParallelVariableOp(const Initializer& init, HTShape global_shape, 
                              const DistributedStates& ds, int64_t local_idx,
                              DataType dtype = kFloat32, bool requires_grad = false,
                              OpMeta op_meta = OpMeta());

Tensor MakeParallelParameterOp(const Initializer& init, HTShape global_shape, 
                               const DistributedStates& ds, int64_t local_idx,
                               DataType dtype = kFloat32, bool requires_grad = false,
                               OpMeta op_meta = OpMeta());                              
} // namespace graph
} // namespace hetu
