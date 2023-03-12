#include "hetu/graph/ops/group.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeGroupOp(OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<GroupOpImpl>(), TensorList(),
                       std::move(op_meta))
    ->out_dep_linker();
}

} // namespace graph
} // namespace hetu
