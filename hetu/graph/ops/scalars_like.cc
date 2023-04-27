#include "hetu/graph/ops/scalars_like.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeScalarsLikeOp(Tensor input, double scalar_value, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ScalarsLikeOpImpl>(scalar_value),
                       {std::move(input)}, std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
