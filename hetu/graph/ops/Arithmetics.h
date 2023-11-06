#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class AddElewiseOpImpl;
class AddByConstOpImpl;

class SubElewiseOpImpl;
class SubByConstOpImpl;
class SubFromConstOpImpl;

class NegateOpImpl;

class MulElewiseOpImpl;
class MulByConstOpImpl;

class DivElewiseOpImpl;
class DivByConstOpImpl;
class DivFromConstOpImpl;

class ReciprocalOpImpl;

class AddElewiseGradientOpImpl;
class SubElewiseGradientOpImpl;
class MulElewiseGradientOpImpl;
class DivElewiseGradientOpImpl;

class AddElewiseOpImpl final: public OpInterface {
 public:
  AddElewiseOpImpl(bool inplace)
  : OpInterface(quote(AddElewiseOp)), _inplace(inplace) {
  }

  inline bool inplace() const{
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AddElewiseOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

class AddByConstOpImpl : public OpInterface {
 public:
  AddByConstOpImpl(double value, bool inplace)
  : OpInterface(quote(AddByConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }
protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AddByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};

class SubElewiseOpImpl : public OpInterface {
 public:
  SubElewiseOpImpl(bool inplace)
  : OpInterface(quote(SubElewiseOp)), _inplace(inplace) {
  }

  inline bool inplace() const{
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubElewiseOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

class SubByConstOpImpl : public OpInterface {
 public:
  SubByConstOpImpl(double value, bool inplace)
  : OpInterface(quote(SubByConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};

class SubFromConstOpImpl : public OpInterface {
  public:
  SubFromConstOpImpl(double value, bool inplace)
  : OpInterface(quote(SubFromConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};

class NegateOpImpl : public OpInterface {
 public:
  NegateOpImpl()
  : OpInterface(quote(NegateOp)) {
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

class MulElewiseOpImpl : public OpInterface {
public:
  MulElewiseOpImpl(bool inplace)
  : OpInterface(quote(MulElewiseOp)), _inplace(inplace) {
  }

  inline bool inplace() const{
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MulElewiseOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

class MulByConstOpImpl : public OpInterface {
 public:
  MulByConstOpImpl(double value, bool inplace)
  : OpInterface(quote(MulByConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MulByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};

class DivElewiseOpImpl : public OpInterface {
 public:
  DivElewiseOpImpl(bool inplace)
  : OpInterface(quote(DivElewiseOp)), _inplace(inplace) {
  }

  inline bool inplace() const{
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivElewiseOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

class DivByConstOpImpl : public OpInterface {
 public:
  DivByConstOpImpl(double value, bool inplace)
  : OpInterface(quote(DivByConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};


class DivFromConstOpImpl : public OpInterface {
 public:
  DivFromConstOpImpl(double value, bool inplace)
  : OpInterface(quote(DivFromConstOp)), _value(value), _inplace(inplace) {
  }

  inline double const_value() const {
    return _value;
  }

  inline bool inplace() const{
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  double _value;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value() && inplace() == rhs_.inplace();
    }
    return false;
  }
};

class ReciprocalOpImpl : public OpInterface {
 public:
  ReciprocalOpImpl()
  : OpInterface(quote(ReciprocalOp)) {
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

class AddElewiseGradientOpImpl : public OpInterface {
 public:
  AddElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : OpInterface(quote(AddElewiseGradientOp)),
  _add_axes(axe),
  _keep_dims(keep_dims),
  _index(index) {
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTAxes _add_axes;

  HTKeepDims _keep_dims;

  int _index;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AddElewiseGradientOpImpl&>(rhs);
      return (index() == rhs_.index() 
              && keep_dims() == rhs_.keep_dims()
              && axes() == rhs_.axes());
    }
    return false;
  }
};

class SubElewiseGradientOpImpl : public OpInterface {
 public:
  SubElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : OpInterface(quote(SubElewiseGradientOp)),
  _add_axes(axe),
  _keep_dims(keep_dims),
  _index(index) {
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTAxes _add_axes;

  HTKeepDims _keep_dims;

  int _index;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubElewiseGradientOpImpl&>(rhs);
      return (index() == rhs_.index() 
              && keep_dims() == rhs_.keep_dims()
              && axes() == rhs_.axes());
    }
    return false;
  }
};

class MulElewiseGradientOpImpl : public OpInterface {
  public:
  MulElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : OpInterface(quote(MulElewiseGradientOp)),
  _add_axes(axe),
  _keep_dims(keep_dims),
  _index(index) {
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTAxes _add_axes;

  HTKeepDims _keep_dims;

  int _index;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MulElewiseGradientOpImpl&>(rhs);
      return (index() == rhs_.index() 
              && keep_dims() == rhs_.keep_dims()
              && axes() == rhs_.axes());
    }
    return false;
  }
};

class DivElewiseGradientOpImpl : public OpInterface {
  public:
  DivElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : OpInterface(quote(DivElewiseGradientOp)),
  _add_axes(axe),
  _keep_dims(keep_dims),
  _index(index) {
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }
  
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTAxes _add_axes;

  HTKeepDims _keep_dims;

  int _index;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivElewiseGradientOpImpl&>(rhs);
      return (index() == rhs_.index() 
              && keep_dims() == rhs_.keep_dims()
              && axes() == rhs_.axes());
    }
    return false;
  }
};

Tensor MakeAddElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivByConstOp(Tensor input, double value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeAddElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstInplaceOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivByConstInplaceOp(Tensor input, double value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstInplaceOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeAddElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeNegateOp(Tensor input, 
                    OpMeta op_meta = OpMeta());

Tensor MakeReciprocalOp(Tensor input, 
                        OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
