#include "hetu/graph/ops/variable.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

bool VariableOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                   StreamIndex stream_id) const {
  if (!OpInterface::DoInstantiate(op, placement, stream_id))
    return false;

  if (_init != nullptr) {
    Graph::AllocVariableData(op->output(0), *_init);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        placement != _provided_data->device()) {
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(_provided_data));
      // TODO: free the provided data in order to save memory
    } else {
      Graph::RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return true;
}

NDArrayList VariableOpImpl::DoAllocOutputs(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& runtime_ctx) const {
  return {Graph::GetVariableData(op->output(0))};
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           init, std::move(shape), dtype, requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           std::move(provided_data), copy_provided_data, dtype,
                           requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

Tensor MakeParameterOp(const Initializer& init, HTShape shape, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(init, std::move(shape), dtype, 
                            requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(std::move(provided_data), copy_provided_data, 
                            dtype, requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

} // namespace graph
} // namespace hetu
