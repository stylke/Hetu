#include "hetu/graph/ops/variable.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

bool VariableOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                   StreamIndex stream_id) const {
  if (!OpInterface::DoInstantiate(op, placement, stream_id))
    return false;

  if (_init != nullptr) {
    auto& data = Graph::GetGraph(op).GetOrAllocVariableData(op->output(0));
    _init->Init(data, 0, stream_id);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        placement != _provided_data->device()) {
      auto& data = Graph::GetGraph(op).GetOrAllocVariableData(op->output(0));
      NDArray::copy(_provided_data, stream_id, data);
      // TODO: free the provided data in order to save memory
    } else {
      Graph::GetGraph(op).RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return true;
}

NDArrayList VariableOpImpl::DoAllocOutputs(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& runtime_ctx) const {
  return {Graph::GetGraph(op).GetOrAllocVariableData(op->output(0))};
}

Tensor MakeVariableOp(const Initializer& init, HTShape shape, DataType dtype,
                      bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           init, std::move(shape), dtype, requires_grad),
                           TensorList(), std::move(op_meta))->output(0);
  out->set_requires_grad(requires_grad);
  return out;
}

Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data,
                      DataType dtype, bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           std::move(provided_data), copy_provided_data, dtype,
                           requires_grad),
                           TensorList(), std::move(op_meta))->output(0);
  out->set_requires_grad(requires_grad);
  return out;
}

Tensor MakeParameterOp(const Initializer& init, HTShape shape, DataType dtype,
                       bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<ParameterOpImpl>(init, std::move(shape),
                           dtype, requires_grad), TensorList(), std::move(op_meta))
                           ->output(0);
  out->set_requires_grad(requires_grad);
  return out;
}

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data,
                       DataType dtype, bool requires_grad, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<ParameterOpImpl>(
                         std::move(provided_data), copy_provided_data, dtype,
                         requires_grad),
                         TensorList(), std::move(op_meta))->output(0);
  out->set_requires_grad(requires_grad);
  return out;
}

} // namespace graph
} // namespace hetu
