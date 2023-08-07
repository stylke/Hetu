#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {


TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
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
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape),
                         {input},
                         std::move(op_meta))->output(0));
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
                         {input},
                         std::move(op_meta))->output(0));
  }
  return std::move(outputs);
}

} // namespace graph
} // namespace hetu
