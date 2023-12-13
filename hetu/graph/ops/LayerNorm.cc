#include "hetu/graph/ops/LayerNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/graph/ops/Reduce.h"

namespace hetu {
namespace graph {

void LayerNormOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  NDArray::layernorm(inputs.at(0), inputs.at(1), inputs.at(2), normalized_shape(),
                     get_eps(), op->instantiation_ctx().stream_index, 
                     outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList LayerNormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  auto grad_input = op->requires_grad(0) ? MakeLayerNormGradientOp(grad_outputs.at(0), op->input(0),
                                          op->input(1), op->output(1), op->output(2),
                                          normalized_shape(), get_eps(), g_op_meta)
                                        : empty;
  return grad_input;
}

HTShapeList LayerNormOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  size_t dim = normalized_shape().size();
  HTShape output_shape = input_shapes.at(0);
  for (size_t i = 0; i < dim; ++i) {
    HT_ASSERT(normalized_shape()[dim - 1 - i] == input_shapes.at(0)[input_shapes.at(0).size() - 1 - i])
    << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
    << normalized_shape() << " and input shape:" << input_shapes.at(0);
    output_shape[input_shapes.at(0).size() - 1 - i] = 1;
  }
  return {input_shapes.at(0), output_shape, output_shape};
}

void LayerNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta) const {
  size_t dim = normalized_shape().size();
  HTShape local_shape = inputs.at(0)->shape();
  int max_dim = local_shape.size() - dim;
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_scale = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_bias = inputs.at(2)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_scale.is_valid() && ds_bias.is_valid()
            && ds_input.get_device_num() == ds_scale.get_device_num()
            && ds_scale.get_device_num() == ds_bias.get_device_num()) 
    << "LayerNormOpDef: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_scale.get_dim(-2) == 1 
            && ds_bias.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(max_dim))
    << "LayerNormOp only support input split in dimension < " << max_dim;
  // scale and bias shape should be normalized_shape, so keep duplicate
  HT_ASSERT(ds_scale.check_pure_duplicate() && ds_bias.check_pure_duplicate())
    << "Scale and bias should be duplicate!";
  outputs.at(0)->set_distributed_states(ds_input); // output
  outputs.at(1)->set_distributed_states(ds_input); // save_mean for backward
  outputs.at(2)->set_distributed_states(ds_input); // save_var for backward
}

void LayerNormGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
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

void LayerNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta) const {
  const DistributedStates& ds_output_grad = inputs.at(0)->get_distributed_states();
  int reduce_dim = inputs.at(0)->ndim() - normalized_shape().size();
  HTAxes axes(reduce_dim);
  HTKeepDims keepdims(reduce_dim);
  for (int d = 0; d < reduce_dim; d++) {
    axes[d] = d;
    keepdims[d] = false;
  }
  DistributedStates ds_bias_scale = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes, keepdims);
  outputs.at(0)->set_distributed_states(ds_output_grad);
  outputs.at(1)->set_distributed_states(ds_bias_scale);
  outputs.at(2)->set_distributed_states(ds_bias_scale);
}

// void LayerNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
//                                              const OpMeta& op_meta) const {
//   const DistributedStates& ds_output_grad = inputs.at(0)->get_distributed_states();
//   int device_num = ds_output_grad.get_device_num();
//   std::unordered_map<int32_t, int32_t> states = ds_output_grad.get_states();
//   std::vector<int32_t> order = ds_output_grad.get_order();
//   std::unordered_map<int32_t, int32_t> res_states;
//   std::vector<int32_t> res_order;
//   int duplicate = ds_output_grad.get_dim(-1);
//   int partial = 1;
//   // 这里给的限制稍微有点严格
//   // case 1: no dup
//   if (duplicate == 1) {
//     for (int i = 0; i < order.size(); i++) {
//       partial *= states[order[i]];
//     }
//     res_states[-2] = partial;
//     res_order.push_back(-2);
//   }
//   // case 2: order: [dup, d1, d2, d3, ...]
//   else if (order[0] == -1) {
//     for (int i = 1; i < order.size(); i++) {
//       partial *= states[order[i]];
//     }
//     res_states[-1] = duplicate;
//     res_order.push_back(-1);
//     if (partial > 1) {
//       res_states[-2] = partial;
//       res_order.push_back(-2);
//     }
//   } 
//   // case 3: order: [d1, dup]
//   else if (order[order.size() - 1] == -1) {
//     HT_ASSERT(order.size() <= 2) 
//       << "LayerNorm: only allow at most 1 dimension splited before dup(-1) in input tensor order!";
//     if (order.size() == 2) {
//       partial *= states[order[0]];
//       res_states[-2] = partial;
//       res_order.push_back(-2);
//     }
//     res_states[-1] = duplicate;
//     res_order.push_back(-1);
//   }
//   // case 4: order: [d1, d2, ..., dup, ..., d3, d4]
//   else {
//     HT_ASSERT(false) 
//       << "LayerNorm: input tensor order is not allowed in [d1, d2, ..., dup, ..., d3, d4] mode, "
//       << "dup shoulbe be in the order[0] or order[-1], or there is no dup.";
//   }

//   DistributedStates ds_bias_scale({device_num, res_states, res_order});
//   outputs.at(0)->set_distributed_states(ds_output_grad);
//   outputs.at(1)->set_distributed_states(ds_bias_scale);
//   outputs.at(2)->set_distributed_states(ds_bias_scale);
// }

TensorList MakeLayerNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias, HTShape normalized_shape, 
                           double eps, OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(bn_scale), std::move(bn_bias)};
  return Graph::MakeOp(
          std::make_shared<LayerNormOpImpl>(normalized_shape, eps),
          std::move(inputs),
          std::move(op_meta))->outputs();   
}

TensorList MakeLayerNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                                   Tensor save_mean, Tensor save_var, HTShape normalized_shape, 
                                   double eps, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<LayerNormGradientOpImpl>(normalized_shape, eps),
          {std::move(output_grad), std::move(input), std::move(bn_scale), 
          std::move(save_mean), std::move(save_var)},
          std::move(op_meta))->outputs();  
}


} // namespace graph
} // namespace hetu
