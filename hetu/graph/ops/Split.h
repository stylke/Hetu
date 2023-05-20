#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SplitOpImpl;
class SplitOp;
class SplitGradientOpImpl;
class SplitGradientOp;

TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta = OpMeta());

TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       OpMeta op_meta = OpMeta());

Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
