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
  AddElewiseOpImpl()
  : OpInterface(quote(AddElewiseOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
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

class AddByConstOpImpl : public OpInterface {
 public:
  AddByConstOpImpl(double value)
  : OpInterface(quote(AddByConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AddByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class SubElewiseOpImpl : public OpInterface {
 public:
  SubElewiseOpImpl()
  : OpInterface(quote(SubElewiseOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
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

class SubByConstOpImpl : public OpInterface {
 public:
  SubByConstOpImpl(double value)
  : OpInterface(quote(SubByConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class SubFromConstOpImpl : public OpInterface {
  public:
  SubFromConstOpImpl(double value)
  : OpInterface(quote(SubFromConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
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
  MulElewiseOpImpl()
  : OpInterface(quote(MulElewiseOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
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

class MulByConstOpImpl : public OpInterface {
 public:
  MulByConstOpImpl(double value)
  : OpInterface(quote(MulByConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MulByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class DivElewiseOpImpl : public OpInterface {
 public:
  DivElewiseOpImpl()
  : OpInterface(quote(DivElewiseOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
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

class DivByConstOpImpl : public OpInterface {
 public:
  DivByConstOpImpl(double value)
  : OpInterface(quote(DivByConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};


class DivFromConstOpImpl : public OpInterface {
 public:
  DivFromConstOpImpl(double value)
  : OpInterface(quote(DivFromConstOp)), _value(value) {
  }

  inline double const_value() const {
    return _value;
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
  
  double _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
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
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

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
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

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
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

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
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

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
