#pragma once

#include "hetu/core/symbol.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ParallelAttentionOpImpl;
class ParallelAttentionOp;
class ParallelAttentionGradientOpImpl;
class ParallelAttentionGradientOp;

class ParallelAttentionOpImpl final : public OpInterface {
 private:
  friend class ParallelAttentionOp;
  struct constructor_access_key {};

 public:
  ParallelAttentionOpImpl(int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                          double p_dropout, double softmax_scale, 
                          bool is_causal, bool return_softmax)
  : OpInterface(quote(ParallelAttentionOp)), 
    _head_dim(head_dim), _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal), _return_softmax(return_softmax) {
  }

  uint64_t op_indicator() const noexcept override {
    return PARALLEL_ATTN_OP;
  } 

  inline double head_dim() const {
    return _head_dim;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

  inline bool return_softmax() const {
    return _return_softmax;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _head_dim;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;
  bool _return_softmax;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ParallelAttentionOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim() &&
             p_dropout() == rhs_.p_dropout() &&
             softmax_scale() == rhs_.softmax_scale() &&
             is_causal() == rhs_.is_causal() &&
             return_softmax() == rhs_.return_softmax();
    } 
    else
      return false;
  }
};

TensorList MakeParallelAttentionOp(Tensor qkv, int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                   double p_dropout = 0.0, double softmax_scale = -1.0, 
                                   bool is_causal = false, bool return_softmax = false, OpMeta op_meta = OpMeta());

class ParallelAttentionGradientOpImpl final : public OpInterface {

 public:
  ParallelAttentionGradientOpImpl(int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                                  double p_dropout, double softmax_scale, bool is_causal)
  : OpInterface(quote(ParallelAttentionGradientOp)), 
    _head_dim(head_dim), _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal) {
  }

  inline double head_dim() const {
    return _head_dim;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _head_dim;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ParallelAttentionGradientOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim()
             && p_dropout() == rhs_.p_dropout()
             && softmax_scale() == rhs_.softmax_scale()
             && is_causal() == rhs_.is_causal();
    } 
    else
      return false;
  }
};

TensorList MakeParallelAttentionGradientOp(Tensor grad_out, Tensor qkv,
                                           Tensor out, Tensor softmax_lse, Tensor rng_state,
                                           int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                           double p_dropout = 0.0, double softmax_scale = -1.0,
                                           bool is_causal = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
