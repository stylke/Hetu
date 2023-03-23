#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, const OpMeta& op_meta) {
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
  return Graph::MakeOp(
        std::make_shared<SliceOpImpl>(begin_pos, output_shape),
        {std::move(input)},
        std::move(op_meta))->output(0);                    
}

} // namespace graph
} // namespace hetu
