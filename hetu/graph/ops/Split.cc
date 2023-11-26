#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/core/symbol.h"

namespace hetu {
namespace graph {

Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta) {
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << op_meta.name << "type 1: use symbolic method";
  HT_ASSERT(input->has_shape());
  // get begin_pos, output_shape
  HT_ASSERT(axes.size() == splits.size());
  int len = axes.size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(axes[i] >= 0);
    HT_ASSERT(splits[i] >= 0);
    HT_ASSERT(indices[i] >= 0 && indices[i] < splits[i]);
  }

  // Split算子在make时，要将输入的tensor设置成symbolic的，之后shape发生改变时，
  // 直接overwrite该tensor中的symbolic shape的value即可，
  // 后续slice算子的shape均会发生改变
  input->init_symbolic_shape(); // leaf
  const SyShape& ori_shape = input->symbolic_shape(); 

  int ndim = ori_shape.size();
  SyShape begin_pos(ndim, 0);
  SyShape output_shape(ori_shape);

  for (int i = 0; i < len; ++i) {
    auto axe = axes[i];
    auto part_size = ori_shape[axe] / splits[i];
    begin_pos[axe] = part_size * indices[i];
    if (indices[i] != splits[i] - 1) {
      output_shape[axe] = part_size;
    } else {
      output_shape[axe] = ori_shape[axe] - begin_pos[axe];
    }
  }

  // 将输出的tensor设置成symbolic的（主要是因为其后可能跟着另一个symbolic算子）
  auto output = Graph::MakeOp(std::make_shared<SliceOpImpl>(std::move(begin_pos), output_shape, -1, false),
                      {std::move(input)}, std::move(op_meta))->output(0);
  output->set_symbolic_shape(std::move(output_shape)); // not leaf
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " split op type 1: finish making";
  return output;
}


// 实现与原版不同？这里只能做到在单一的dim上的切分
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " split op type 2: " 
    << "input_shape = " << input->shape() << " and num_chunks = " << num_chunks;
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());

  input->init_symbolic_shape(); // leaf
  const SyShape& ori_shape = input->symbolic_shape(); 

  auto chunk_size = ori_shape[dim] / num_chunks;
  auto chunk_sum = Symbol<int64_t>(0);
  TensorList outputs = {};

  for (int i = 0; i < num_chunks; ++i) {
    SyShape begin_pos(input->ndim(), 0);
    SyShape output_shape(ori_shape);
    output_shape[dim] = i == num_chunks - 1 ? (ori_shape[dim] - 1) % chunk_size + 1
                                            : chunk_size;
    begin_pos[dim] = chunk_sum;
    chunk_sum = chunk_sum + chunk_size;
    HT_LOG_DEBUG << "ckpt1";
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(std::move(begin_pos), output_shape, -1, false),
                         {input}, op_meta)->output(0));
    HT_LOG_DEBUG << "ckpt2";
    outputs[i]->set_symbolic_shape(std::move(output_shape));
    HT_LOG_DEBUG << "ckpt3";
  }
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " split op type 2: finish making";
  return outputs;
}

// deprecated: only used in gpt inference, before symbolic shape is realized
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       int64_t padding_axis, OpMeta op_meta) {
  HT_LOG_WARN << "This method is almost deprecated";
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  padding_axis = NDArrayMeta::ParseAxis(padding_axis, input->ndim());
  HT_ASSERT(dim != padding_axis) << "Split dim can't be the padding dim.";
  int64_t chunk_sum = 0;
  chunk_sum = 0;
  int64_t chunk_size = DIVUP(input->shape(dim), num_chunks);
  HTShape begin_pos(input->ndim());
  HTShape output_shape = input->shape();
  TensorList outputs = {};
  for (int i = 0; i < num_chunks; ++i) {
    output_shape[dim] = i == num_chunks - 1 ? (input->shape(dim) - 1) % chunk_size + 1
                                            : chunk_size;
    begin_pos[dim] = chunk_sum;
    chunk_sum += chunk_size;
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape, padding_axis, false),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

// seems deprecated
TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_RUNTIME_ERROR << "This method is deprecated";
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  int64_t chunk_sum = 0;
  int64_t len = chunks.size();
  for (int i = 0; i < len; ++i) {
    chunk_sum += chunks[i];
  }
  HT_ASSERT(chunk_sum == input->shape(dim));
  chunk_sum = 0;
  HTShape begin_pos(input->ndim());
  HTShape output_shape = input->shape();
  TensorList outputs = {};
  for (int i = 0; i < len; ++i) {
    output_shape[dim] = chunks[i];
    begin_pos[dim] = chunk_sum;
    chunk_sum += chunks[i];
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape, -1, false),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

} // namespace graph
} // namespace hetu
