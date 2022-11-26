#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class MaxPoolOpDef;
class MaxPoolOp;
class MaxPoolGradientOpDef;
class MaxPoolGradientOp;

class MaxPoolOpDef : public OperatorDef {
 private:
  friend class MaxPoolOp;
  struct constrcutor_access_key {};

 public:
  MaxPoolOpDef(const constrcutor_access_key&, Tensor input, size_t kernel_H,
               size_t kernel_W, size_t padding, size_t stride,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MaxPoolOp), {input}, op_meta),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {
    HTShape shape = {-1, -1, -1, -1};
    if (input->has_shape()) {
      int64_t N = input->shape(0);
      int64_t C = input->shape(1);
      int64_t H = input->shape(2);
      int64_t W = input->shape(3);
      int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
      int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
      shape = {N, C, p_H, p_W};
    }
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
  }

  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _kernel_H;

  size_t _kernel_W;

  size_t _padding;

  size_t _stride;
};

class MaxPoolOp final : public OpWrapper<MaxPoolOpDef> {
 public:
  MaxPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, size_t padding,
            size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MaxPoolOpDef>(
      make_ptr<MaxPoolOpDef>(MaxPoolOpDef::constrcutor_access_key(), input,
                             kernel_H, kernel_W, padding, stride, op_meta)) {}
};

class MaxPoolGradientOpDef : public OperatorDef {
 private:
  friend class MaxPoolGradientOp;
  struct constrcutor_access_key {};

 public:
  MaxPoolGradientOpDef(const constrcutor_access_key&, Tensor output,
                       Tensor output_grad, Tensor input, size_t kernel_H,
                       size_t kernel_W, size_t padding, size_t stride,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MaxPoolGradientOp), {output, output_grad, input},
                op_meta),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {
    AddOutput(input->meta());
  }

  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _kernel_H;

  size_t _kernel_W;

  size_t _padding;

  size_t _stride;
};

class MaxPoolGradientOp final : public OpWrapper<MaxPoolGradientOpDef> {
 public:
  MaxPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                    size_t kernel_H, size_t kernel_W, size_t padding,
                    size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MaxPoolGradientOpDef>(make_ptr<MaxPoolGradientOpDef>(
      MaxPoolGradientOpDef::constrcutor_access_key(), output, output_grad,
      input, kernel_H, kernel_W, padding, stride, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
