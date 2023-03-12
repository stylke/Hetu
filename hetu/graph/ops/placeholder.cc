#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakePlaceholderOp(NDArrayMeta data_meta, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<PlaceholderOpImpl>(std::move(data_meta)),
           TensorList(), std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
