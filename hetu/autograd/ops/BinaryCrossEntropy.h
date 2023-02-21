#pragma once

#include "hetu/autograd/operator.h"

#include "hetu/impl/communication/comm_group.h"
using namespace hetu::impl::comm;

namespace hetu {
namespace autograd {

class BinaryCrossEntropyOpDef;
class BinaryCrossEntropyOp;
class BinaryCrossEntropyGradientOpDef;
class BinaryCrossEntropyGradientOp;

class BinaryCrossEntropyOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyOpDef(const constrcutor_access_key&, Tensor preds,
                          Tensor labels, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyOp), {preds, labels}, op_meta) {
    AddOutput(preds->meta());
    ForwardDeduceStates();
    // test: 这里假设每个op的output只有一个, 简单测试输出的ds, 如果op的output不只一个, 则只输出第0个的情况
    auto ds = _outputs[0]->get_distributed_states();
    auto local_device = GetLocalDevice();
    HT_LOG_DEBUG << local_device << ": " << name() << " definition, output[0] states: " << ds.print_states() << ", shape: " << _outputs[0]->shape();     
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BinaryCrossEntropyOp final : public OpWrapper<BinaryCrossEntropyOpDef> {
 public:
  BinaryCrossEntropyOp(Tensor preds, Tensor labels,
                       const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyOpDef>(make_ptr<BinaryCrossEntropyOpDef>(
      BinaryCrossEntropyOpDef::constrcutor_access_key(), preds, labels,
      op_meta)) {}
};

class BinaryCrossEntropyGradientOpDef : public OperatorDef {
 private:
  friend class BinaryCrossEntropyGradientOp;
  struct constrcutor_access_key {};

 public:
  BinaryCrossEntropyGradientOpDef(const constrcutor_access_key&, Tensor preds,
                                  Tensor labels, Tensor grad_output,
                                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BinaryCrossEntropyGradientOp),
                {preds, labels, grad_output}, op_meta) {
    AddOutput(preds->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class BinaryCrossEntropyGradientOp final
: public OpWrapper<BinaryCrossEntropyGradientOpDef> {
 public:
  BinaryCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               const OpMeta& op_meta = OpMeta())
  : OpWrapper<BinaryCrossEntropyGradientOpDef>(
      make_ptr<BinaryCrossEntropyGradientOpDef>(
        BinaryCrossEntropyGradientOpDef::constrcutor_access_key(), preds,
        labels, grad_output, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
