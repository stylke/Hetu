#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  // get begin_pos, output_shape
  HT_ASSERT(axes.size() == splits.size());
  int len = axes.size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(axes[i] >= 0);
    HT_ASSERT(splits[i] >= 0);
    HT_ASSERT(indices[i] >= 0 && indices[i] < splits[i]);
  }

  HTShape ori_shape = input->shape();
  int ndim = ori_shape.size();
  HTShape begin_pos(ndim);
  HTShape output_shape(ndim);
  for (int i = 0; i < ndim; ++i) {
    begin_pos[i] = 0;
    output_shape[i] = ori_shape[i];
  }
  for (int i = 0; i < len; ++i) {
    int64_t axe = axes[i];
    int64_t ind = indices[i];
    int64_t spl = splits[i];
    int64_t part_size = ori_shape[axe] / spl;
    begin_pos[axe] = ind * part_size;
    if (ind != spl - 1) {
      output_shape[axe] = part_size;
    } else {
      output_shape[axe] = ori_shape[axe] - begin_pos[axe];
    }
  }
  return Graph::MakeOp(std::make_shared<SliceOpImpl>(std::move(begin_pos), std::move(output_shape)),
                      {std::move(input)}, std::move(op_meta))->output(0);
}


// 实现与原版不同？这里只能做到在单一的dim上的切分
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  int64_t chunk_sum = 0;
  chunk_sum = 0;
  int64_t chunk_size = input->shape(dim) / num_chunks;
  HTShape begin_pos(input->ndim());
  HTShape output_shape = input->shape();
  TensorList outputs = {};
  for (int i = 0; i < num_chunks; ++i) {
    output_shape[dim] = i == num_chunks - 1 ? (input->shape(dim) - 1) % chunk_size + 1
                                            : chunk_size;
    begin_pos[dim] = chunk_sum;
    chunk_sum += chunk_size;
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       OpMeta op_meta) {
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
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

} // namespace graph
} // namespace hetu
