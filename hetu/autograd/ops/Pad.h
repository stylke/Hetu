#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class PadOpDef;
class PadOp;
class PadGradientOpDef;
class PadGradientOp;

class PadOpDef : public OperatorDef {
 private:
  friend class PadOp;
  struct constrcutor_access_key {};

 public:
  PadOpDef(const constrcutor_access_key&, Tensor input, const HTShape& paddings,
           const std::string& mode, double constant, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(PadOp), {input}, op_meta),
    _mode(mode),
    _paddings(paddings),
    _constant(constant) {
    HTShape shape;
    if (input->has_shape()) {
      shape = input->shape();
      size_t len = paddings.size();
      for (size_t i = 0; i < 4; ++i) {
        if (i >= (4 - len / 2)) {
          shape[i] = shape[i] + paddings[(i - (4 - len / 2)) * 2] +
            paddings[(i - (4 - len / 2)) * 2 + 1];
        }
      }
    }
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

  const std::string& get_mode() const {
    return _mode;
  }

  HTShape get_paddings() const {
    return _paddings;
  }

  double get_constant() const {
    return _constant;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTShape _paddings;

  double _constant;
};

class PadOp final : public OpWrapper<PadOpDef> {
 public:
  PadOp(Tensor input, const HTShape& paddings, std::string mode, double constant,
        const OpMeta& op_meta = OpMeta())
  : OpWrapper<PadOpDef>(make_ptr<PadOpDef>(PadOpDef::constrcutor_access_key(),
                                           input, paddings, mode, constant,
                                           op_meta)) {}
};

class PadGradientOpDef : public OperatorDef {
 private:
  friend class PadGradientOp;
  struct constrcutor_access_key {};

 public:
  PadGradientOpDef(const constrcutor_access_key&, Tensor grad_output,
                   const HTShape& paddings, const std::string& mode,
                   const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(PadGradientOp), {grad_output}, op_meta),
    _mode(mode),
    _paddings(paddings) {
    HTShape shape = grad_output->shape();
    size_t len = paddings.size();
    for (size_t i = 0; i < 4; ++i) {
      if (i >= (4 - len / 2)) {
        shape[i] = shape[i] - paddings[(i - (4 - len / 2)) * 2] -
          paddings[(i - (4 - len / 2)) * 2 + 1];
      }
    }
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

  const std::string& get_mode() const {
    return _mode;
  }

  HTShape get_paddings() const {
    return _paddings;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _mode;

  HTShape _paddings;
};

class PadGradientOp final : public OpWrapper<PadGradientOpDef> {
 public:
  PadGradientOp(Tensor grad_output, const HTShape& paddings, std::string mode,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<PadGradientOpDef>(
      make_ptr<PadGradientOpDef>(PadGradientOpDef::constrcutor_access_key(),
                                 grad_output, paddings, mode, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
