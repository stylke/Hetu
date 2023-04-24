#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class ConcatenateOpDef;
class ConcatenateOp;
class ConcatenateGradientOpDef;
class ConcatenateGradientOp;

class ConcatenateOpDef : public OperatorDef {
 private:
  friend class ConcatenateOp;
  struct constrcutor_access_key {};

 public:
  ConcatenateOpDef(const constrcutor_access_key&, const TensorList& inputs,
                   size_t axis = 0, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatenateOp), inputs, op_meta), _axis(axis) {
    int len = inputs.size();
    grad_inputs.resize(len);

    bool flag = true;
    for (int i = 0; i < len; ++i) {
      if (!inputs.at(i)->has_shape()) {
        flag = false;
        break;
      }
    }
    HTShape out_shape = {};
    if (flag) {
      out_shape = inputs.at(0)->shape();
      int n_dim = out_shape.size();
      int out_dim = out_shape[axis];
      int ind = 0;
      ind += 1;
      for (int i = 1; i < len; ++i) {
        HTShape shape = inputs.at(i)->shape();
        HT_ASSERT(shape.size() == out_shape.size());
        for (int j = 0; j < n_dim; ++j) {
          if (j != (int) axis) {
            HT_ASSERT(shape[j] == out_shape[j] || shape[j] == -1 ||
                      out_shape[j] == -1);
          } else {
            ind += 1;
            out_dim += shape[j];
          }
        }
      }
      out_shape[axis] = out_dim;
    }
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(out_shape));
    if (op_meta.is_deduce_states) {
      DeduceStates();
    }
  }

  void DeduceStates() override;

  size_t get_axis() const {
    return _axis;
  }

  std::vector<ConcatenateGradientOp> grad_inputs;

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;
};

class ConcatenateOp final : public OpWrapper<ConcatenateOpDef> {
 public:
  ConcatenateOp(const TensorList& inputs, size_t axis = 0,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatenateOpDef>(make_ptr<ConcatenateOpDef>(
      ConcatenateOpDef::constrcutor_access_key(), inputs, axis, op_meta)) {}
};

class ConcatenateGradientOpDef : public OperatorDef {
 private:
  friend class ConcatenateGradientOp;
  struct constrcutor_access_key {};

 public:
  ConcatenateGradientOpDef(const constrcutor_access_key&, Tensor input,
                           Tensor grad_output, size_t axis,
                           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatenateGradientOp), {input, grad_output}, op_meta),
    _axis(axis) {
    AddOutput(input->meta());
    DeduceStates();
  }

  void DeduceStates() override;

  size_t get_axis() const {
    return _axis;
  }

  size_t get_offset() const {
    return _offset;
  }

  void set_offset(size_t offset) {
    _offset = offset;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;

  size_t _offset;
};

class ConcatenateGradientOp final : public OpWrapper<ConcatenateGradientOpDef> {
 public:
  ConcatenateGradientOp() : OpWrapper<ConcatenateGradientOpDef>() {}
  ConcatenateGradientOp(Tensor input, Tensor grad_output, size_t axis,
                        const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatenateGradientOpDef>(make_ptr<ConcatenateGradientOpDef>(
      ConcatenateGradientOpDef::constrcutor_access_key(), input, grad_output,
      axis, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
