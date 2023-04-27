#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/headers.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {
namespace graph {

bool DataH2DOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                  StreamIndex stream_id) const {
  if (!placement.is_cuda())
    return false;
  HT_RUNTIME_ERROR_IF(!op->input(0)->placement().is_cpu())
    << "Failed to instantiate op " << op << ": "
    << "The input \"" << op->input(0) << "\" must be placed on "
    << "a host device. Got " << op->input(0)->placement() << ".";
  auto& inst_ctx = op->instantiation_ctx();
  inst_ctx.placement = placement;
  inst_ctx.stream_index = stream_id;
  inst_ctx.start = std::make_unique<hetu::impl::CUDAEvent>(inst_ctx.placement);
  inst_ctx.stop = std::make_unique<hetu::impl::CUDAEvent>(inst_ctx.placement);
  op->output(0)->set_placement(placement);
  return true;
}

TensorList DataH2DOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  return {MakeDataD2HOp(device(), grad_outputs.front(),
                        op->grad_op_meta().set_name(op->grad_name()))};
}

void DataH2DOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& runtime_ctx) const {
  NDArray::to(inputs.front(), outputs.front()->device(),
              outputs.front()->dtype(), op->instantiation_ctx().stream_index,
              outputs.front());
}

bool DataD2HOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                  StreamIndex stream_id) const {
  if (!placement.is_cpu())
    return false;
  HT_RUNTIME_ERROR_IF(!op->input(0)->placement().is_cuda())
    << "Failed to instantiate op " << op << ": "
    << "The input \"" << op->input(0) << "\" must be placed on "
    << "a CUDA device. Got " << op->input(0)->placement() << ".";
  auto& inst_ctx = op->instantiation_ctx();
  inst_ctx.placement = placement;
  inst_ctx.stream_index = stream_id;
  inst_ctx.start = std::make_unique<hetu::impl::CUDAEvent>(inst_ctx.placement);
  inst_ctx.stop = std::make_unique<hetu::impl::CUDAEvent>(inst_ctx.placement);
  op->output(0)->set_placement(placement);
  return true;
}

TensorList DataD2HOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  return {MakeDataD2HOp(device(), grad_outputs.front(),
                        op->grad_op_meta().set_name(op->grad_name()))};
}

void DataD2HOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& runtime_ctx) const {
  NDArray::to(inputs.front(), outputs.front()->device(),
              outputs.front()->dtype(), op->instantiation_ctx().stream_index,
              outputs.front());
}

Tensor MakeDataH2DOp(Device device, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<DataH2DOpImpl>(std::move(device)),
                       {std::move(input)}, std::move(op_meta))
    ->output(0);
}

Tensor MakeDataD2HOp(Device device, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<DataD2HOpImpl>(std::move(device)),
                       {std::move(input)}, std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
