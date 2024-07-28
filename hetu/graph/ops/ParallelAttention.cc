#include "hetu/graph/ops/ParallelAttention.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

static int64_t get_local_seq_len(const Tensor& input, const SyShapeList& multi_seq_lens_symbol) {
  auto& graph = input->producer()->graph();
  HT_ASSERT(multi_seq_lens_symbol.size() == graph.NUM_STRATEGY 
            && graph.CUR_STRATEGY_ID < multi_seq_lens_symbol.size())
    << "strategy num should be matched";
  HT_ASSERT(!input->cur_ds_union().is_hetero() 
            || input->cur_ds_union().size() == multi_seq_lens_symbol.at(graph.CUR_STRATEGY_ID).size())
    << "ds union size and seq lens symbol size should be matched";
  auto& cur_seq_lens_symbol = multi_seq_lens_symbol[graph.CUR_STRATEGY_ID];
  if (graph.USE_HETERO_ID) {
    return cur_seq_lens_symbol.at(graph.CUR_HETERO_ID)->get_val();
  } else {
    if (input->cur_ds_union().is_hetero()) {
      // inferred_local_placement_group_idx sucks!
      auto idx = input->producer()->inferred_local_placement_group_idx();
      return cur_seq_lens_symbol.at(idx)->get_val();
    } else {
      for (auto& symbol : cur_seq_lens_symbol) {
        HT_ASSERT(symbol->get_val() == cur_seq_lens_symbol.at(0)->get_val())
          << "all seq lens should be equal in homo setting";
      }
      return cur_seq_lens_symbol.at(0)->get_val();
    }
  }
}

std::vector<NDArrayMeta> ParallelAttentionOpImpl::DoInferMeta(const TensorList& inputs) const {
  std::vector<NDArrayMeta> out_metas = {};
  auto& input = inputs.at(0); // packed qkv
  NDArrayMeta base = input->meta();
  HT_ASSERT(input->shape().size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, num_head * head_dim * 3]";
  int64_t batch_size_mul_seq_len = input->shape(0);
  int64_t num_heads_mul_head_dim = input->shape(1);
  // workaround 
  // 这里不得不使用CUR_HETERO_ID（就算外部没有进行USE_HETERO_ID）
  // 在define graph中，该值一定是0
  // 在exec graph中，该值会在Instantiate中MakeOp前被合理地设置
  input->producer()->graph().USE_HETERO_ID = true;
  int64_t seq_len = get_local_seq_len(input, _multi_seq_lens_symbol);
  input->producer()->graph().USE_HETERO_ID = false;
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len"
    << ", but found dim 0 is " << batch_size_mul_seq_len
    << ", and seq len is " << seq_len;
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % (_head_dim * 3) == 0)
    << "packed qkv dim 1 should be divided by head dim * 3";
  int64_t num_heads = num_heads_mul_head_dim / (_head_dim * 3);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  // TODO: support padding
  out_metas.emplace_back(base.set_shape({batch_size_mul_seq_len, num_heads_mul_head_dim / 3})); // out
  out_metas.emplace_back(base.set_shape({batch_size, num_heads, seq_len}).set_dtype(kFloat)); // softmax_lse
  out_metas.emplace_back(base.set_shape({2}).set_device(kCPU).set_dtype(kInt64)); // rng_state
  return out_metas;
}

void ParallelAttentionOpImpl::DoCompute(Operator& op, 
                                        const NDArrayList& inputs, NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  /*
  // deprecated version: output splitted q, k, v as well
  HT_ASSERT(op->input(0)->num_consumers() == 1)
    << "qkv should only consumed by the ParallelAttentionOp"
    << ", so that it will be released but the splitted output (q, k, v) remains";
  */
  // 拿到qkv先做reshape和split
  // runtime时一定要从ctx中取shape信息
  const auto& qkv = inputs.at(0);
  auto& out = outputs.at(0);
  auto& softmax_lse = outputs.at(1);
  auto& rng_state = outputs.at(2);
  HTShape input_shape = ctx.get_runtime_shape(op->input(0)->id());
  HT_ASSERT(input_shape.size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
  int64_t batch_size_mul_seq_len = input_shape.at(0);
  int64_t num_heads_mul_head_dim = input_shape.at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len";
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % (_head_dim * 3) == 0)
    << "packed qkv dim 1 should be divided by head dim * 3";
  int64_t num_heads = num_heads_mul_head_dim / (_head_dim * 3);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  auto stream_index = op->instantiation_ctx().stream_index;
  auto reshaped_qkv = NDArray::reshape(qkv, {batch_size, seq_len, num_heads, 3 * _head_dim}, stream_index);
  auto reshaped_out = NDArray::reshape(out, {batch_size, seq_len, num_heads, _head_dim}, stream_index);
  // self-attn
  HTShape q_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape k_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape v_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape q_begin_pos = {0, 0, 0, 0};
  HTShape k_begin_pos = {0, 0, 0, _head_dim};
  HTShape v_begin_pos = {0, 0, 0, 2 * _head_dim};
  auto q = NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_index);
  auto k = NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_index);
  auto v = NDArray::slice(reshaped_qkv, v_begin_pos, v_shape, stream_index);
  HT_LOG_DEBUG << "[ParallelAttn]: q (same as k, v) shape is " << q->shape();
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(_head_dim, -0.5);
  // TODO: ring attn
  NDArray empty_ndarray = NDArray();
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttn,
                               q, k, v, reshaped_out, empty_ndarray,
                               empty_ndarray, empty_ndarray, empty_ndarray, softmax_lse,
                               empty_ndarray, rng_state, p_dropout(), softmax_scale_,
                               is_causal(), return_softmax(), op->instantiation_ctx().stream());
}

TensorList ParallelAttentionOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList empty = {Tensor()};
  return op->requires_grad(0) ? MakeParallelAttentionGradientOp(grad_outputs.at(0), op->input(0),
                                                                op->output(0), op->output(1), op->output(2), 
                                                                _head_dim, _multi_seq_lens_symbol, _multi_cp_group_symbol, 
                                                                p_dropout(), softmax_scale(), is_causal(), op->grad_op_meta().set_name(op->grad_name()))
                              : empty;
}

HTShapeList ParallelAttentionOpImpl::DoInferShape(Operator& op, 
                                                  const HTShapeList& input_shapes, 
                                                  RuntimeContext& ctx) const {
  HTShapeList out_shapes;
  HT_ASSERT(input_shapes.at(0).size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
  int64_t batch_size_mul_seq_len = input_shapes.at(0).at(0);
  int64_t num_heads_mul_head_dim = input_shapes.at(0).at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len";
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % (_head_dim * 3) == 0)
    << "packed qkv dim 1 should be divided by head dim * 3";
  int64_t num_heads = num_heads_mul_head_dim / (_head_dim * 3);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  out_shapes.emplace_back(HTShape{batch_size_mul_seq_len, num_heads_mul_head_dim / 3}); // out
  out_shapes.emplace_back(HTShape{batch_size, num_heads, seq_len}); // softmax_lse
  out_shapes.emplace_back(HTShape{2}); // rng_state
  return out_shapes;
}

std::vector<NDArrayMeta> ParallelAttentionGradientOpImpl::DoInferMeta(const TensorList& inputs) const {
  HT_ASSERT(inputs.at(1)->shape().size() == 2)
    << "ParallelAttentionGradientOp input 1 shape should be [batch_size * seq_len, num_heads * head_dim * 3]";
  return {inputs.at(1)->meta()};
}

void ParallelAttentionGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                                NDArrayList& outputs, RuntimeContext& ctx) const {
  const auto& grad_output = inputs.at(0);
  const auto& qkv = inputs.at(1);
  const auto& out = inputs.at(2);
  const auto& softmax_lse = inputs.at(3);
  const auto& rng_state = inputs.at(4);
  auto& grad_input = outputs.at(0);
  HTShape grad_output_shape = ctx.get_runtime_shape(op->input(0)->id());
  HT_ASSERT(grad_output_shape.size() == 2)
    << "ParallelAttentionGradientOp only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
  int64_t batch_size_mul_seq_len = grad_output_shape.at(0);
  int64_t num_heads_mul_head_dim = grad_output_shape.at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "grad output dim 0 should be divided by seq len";
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % _head_dim == 0)
    << "grad output dim 1 should be divided by head dim";
  int64_t num_heads = num_heads_mul_head_dim / _head_dim;
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention backward only supports head dimension at most 256 and must be divided by 8";
  auto stream_index = op->instantiation_ctx().stream_index;
  auto reshaped_qkv = NDArray::reshape(qkv, {batch_size, seq_len, num_heads, 3 * _head_dim}, stream_index);
  auto reshaped_out = NDArray::reshape(out, {batch_size, seq_len, num_heads, _head_dim}, stream_index);
  auto reshaped_grad_output = NDArray::reshape(grad_output, {batch_size, seq_len, num_heads, _head_dim}, stream_index);
  auto reshaped_grad_input = NDArray::reshape(grad_input, {batch_size, seq_len, num_heads, 3 * _head_dim}, stream_index);
  HTShape q_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape k_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape v_shape = {batch_size, seq_len, num_heads, _head_dim};
  HTShape q_begin_pos = {0, 0, 0, 0};
  HTShape k_begin_pos = {0, 0, 0, _head_dim};
  HTShape v_begin_pos = {0, 0, 0, 2 * _head_dim};
  auto q = NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_index);
  auto k = NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_index);
  auto v = NDArray::slice(reshaped_qkv, v_begin_pos, v_shape, stream_index);
  auto dq = NDArray::slice(reshaped_grad_input, q_begin_pos, q_shape, stream_index);
  auto dk = NDArray::slice(reshaped_grad_input, q_begin_pos, q_shape, stream_index);
  auto dv = NDArray::slice(reshaped_grad_input, q_begin_pos, q_shape, stream_index);
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(_head_dim, -0.5);
  // TODO: ring attn
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::FlashAttnGradient, reshaped_grad_output,
                               q, k, v, reshaped_out,
                               const_cast<NDArray&>(softmax_lse), const_cast<NDArray&>(rng_state), 
                               dq, dk, dv, p_dropout(), softmax_scale_,
                               is_causal(), op->instantiation_ctx().stream());
  // flash-attn already supports uncontiguous outputs
  /*
  // concat dq, dk, dv to reshaped_grad_input
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Concatenate, dq,
                               reshaped_grad_input, 3, 0, op->instantiation_ctx().stream());
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Concatenate, dk,
                               reshaped_grad_input, 3, _head_dim, op->instantiation_ctx().stream());
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Concatenate, dv,
                               reshaped_grad_input, 3, 2 * _head_dim, op->instantiation_ctx().stream());
  */
}

HTShapeList ParallelAttentionGradientOpImpl::DoInferShape(Operator& op, 
                                                          const HTShapeList& input_shapes, 
                                                          RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(1).size() == 2)
    << "ParallelAttentionGradientOp input 1 shape should be [batch_size * seq_len, num_heads * head_dim * 3]";
  return {input_shapes.at(1)};
}

TensorList MakeParallelAttentionOp(Tensor qkv, int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                   double p_dropout, double softmax_scale, 
                                   bool is_causal, bool return_softmax, OpMeta op_meta) {
  TensorList inputs = {std::move(qkv)};
  return Graph::MakeOp(
    std::make_shared<ParallelAttentionOpImpl>(head_dim, std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol), 
      p_dropout, softmax_scale, is_causal, return_softmax),
    std::move(inputs),
    std::move(op_meta))->outputs();
}

TensorList MakeParallelAttentionGradientOp(Tensor grad_out, Tensor qkv, 
                                           Tensor out, Tensor softmax_lse, Tensor rng_state,
                                           int64_t head_dim, SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                                           double p_dropout, double softmax_scale,
                                           bool is_causal, OpMeta op_meta) {
  TensorList inputs = {std::move(grad_out), std::move(qkv), std::move(out), std::move(softmax_lse), std::move(rng_state)};
  return Graph::MakeOp(
    std::make_shared<ParallelAttentionGradientOpImpl>(head_dim,  std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol), 
      p_dropout, softmax_scale, is_causal),
    std::move(inputs),
    std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
