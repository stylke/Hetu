#include "hetu/graph/ops/LayerNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LayerNormOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
  //                              hetu::impl::LayerNorm, inputs.at(0),
  //                              inputs.at(1), inputs.at(2), outputs.at(1), outputs.at(2),
  //                              outputs.at(0), get_eps(), op->instantiation_ctx().stream());
  NDArray::layernorm(inputs.at(0), inputs.at(1), inputs.at(2), normalized_shape(),
                     get_eps(), op->instantiation_ctx().stream_index, 
                     outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList LayerNormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  auto grad_input = op->require_grad(0) ? MakeLayerNormGradientOp(grad_outputs.at(0), op->input(0),
                                          op->input(1), op->output(1), op->output(2),
                                          normalized_shape(), get_eps(), g_op_meta)
                                        : empty;
  return grad_input;
}

HTShapeList LayerNormOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  HTShape local_shape = input_shapes.at(0);
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  return {input_shapes.at(0), local_shape, local_shape};
}

void LayerNormGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::LayerNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), inputs.at(3), inputs.at(4), normalized_shape().size(),
    get_eps(), op->instantiation_ctx().stream());
}

HTShapeList
LayerNormGradientOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(2)};
}

TensorList MakeLayerNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias, HTShape normalized_shape, 
                           double eps, const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<LayerNormOpImpl>(normalized_shape, eps),
          {std::move(input), std::move(bn_scale), std::move(bn_bias)},
          std::move(op_meta))->outputs();   
}

TensorList MakeLayerNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                                   Tensor save_mean, Tensor save_var, HTShape normalized_shape, 
                                   double eps, const OpMeta& op_meta) {
  return Graph::MakeOp(
          std::make_shared<LayerNormOpImpl>(normalized_shape, eps),
          {std::move(output_grad), std::move(input), std::move(bn_scale), 
          std::move(save_mean), std::move(save_var)},
          std::move(op_meta))->outputs();  
}


} // namespace graph
} // namespace hetu
