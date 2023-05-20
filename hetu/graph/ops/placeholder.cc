#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

// in_degree=0, should set distributed states manually
Tensor MakePlaceholderOp(NDArrayMeta data_meta, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<PlaceholderOpImpl>(std::move(data_meta)),
           TensorList(), std::move(op_meta.set_is_deduce_states(false)))
    ->output(0);
}

} // namespace graph
} // namespace hetu
