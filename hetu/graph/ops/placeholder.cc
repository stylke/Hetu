#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

// in_degree=0, should set distributed states manually
Tensor MakePlaceholderOp(NDArrayMeta data_meta, const DistributedStates& ds, OpMeta op_meta) {
  Tensor output = Graph::MakeOp(
    std::make_shared<PlaceholderOpImpl>(std::move(data_meta)),
    TensorList(), std::move(op_meta.set_is_deduce_states(false)))
    ->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1) 
      << "DistributedStates for PlaceholderOp must be valid! got: " 
      << ds.ds_info();
    output->set_distributed_states(ds);
  }
  return output;
}

} // namespace graph
} // namespace hetu
