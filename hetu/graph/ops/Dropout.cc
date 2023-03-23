// #include "hetu/graph/ops/Dropout.h"
// #include "hetu/graph/headers.h"
// #include "hetu/graph/ops/kernel_links.h"
// #include "hetu/impl/random/CPURandomState.h"

// namespace hetu {
// namespace graph {

// NDArrayList DropoutOpImpl::DoCompute(Operator& op,
//                                      const NDArrayList& inputs,
//                                      RuntimeContext& ctx) const {
//   NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
//   if (recompute()) {
//     uint64_t seed = hetu::impl::GenNextRandomSeed();
//     ctx.get_op_ctx(id()).put_uint64("seed", seed);
//     HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
//                                  hetu::impl::Dropout, inputs.at(0),
//                                  1 - keep_prob(), seed, outputs[0], op->instantiation_ctx().stream());
//     return outputs;
//   } else {
//     HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
//                                  hetu::impl::Dropout, inputs.at(0),
//                                  1 - keep_prob(), 0, outputs[0], op->instantiation_ctx().stream());
//     return outputs;
//   }
// }

// TensorList DropoutOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
//   if (recompute()) {
//     return {DropoutGradientWithRecomputationOp(
//               grad_outputs.at(0), id(), keep_prob(), op->grad_op_meta().set_name(op->grad_name()))
//               ->output(0)};
//   } else {
//     return {DropoutGradientOp(grad_outputs.at(0), output(0), keep_prob(),
//                               op->grad_op_meta().set_name(op->grad_name()))
//               ->output(0)};
//   }
// }

// void DropoutOpImpl::DoInferMeta() {
//   AddOutput(op->input(0)->meta());
// }

// HTShapeList DropoutOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
//   return {input_shapes.at(0)};
// }

// NDArrayList DropoutGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
//                                             RuntimeContext& ctx) const {
//   NDArrayList outputs = DoAllocOutputs(inputs, ctx);
//   HT_DISPATCH_KERNEL_CUDA_ONLY(
//     op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradient, inputs.at(0),
//     inputs.at(1), 1 - keep_prob(), outputs[0], op->instantiation_ctx().stream());
//   return outputs;
// }

// void DropoutGradientOpImpl::DoInferMeta() {
//   AddOutput(op->input(0)->meta());
// }

// HTShapeList
// DropoutGradientOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
//   CheckNumInputsEqual(input_shapes.size());
//   return {input_shapes.at(0)};
// }

// NDArrayList
// DropoutGradientWithRecomputationOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
//                                                  RuntimeContext& ctx) const {
//   NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(inputs, ctx);
//   uint64_t seed = ctx.get_op_ctx(_forward_op).get_uint64("seed");
//   HT_DISPATCH_KERNEL_CUDA_ONLY(
//     op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradientWithRecomputation,
//     inputs.at(0), 1 - keep_prob(), seed, outputs[0], op->instantiation_ctx().stream());
//   return outputs;
// }

// void DropoutGradientWithRecomputationOpImpl::DoInferMeta() {
//   AddOutput(op->input(0)->meta());
// }

// HTShapeList DropoutGradientWithRecomputationOpImpl::DoInferShape(
//   const HTShapeList& input_shapes) {
//   CheckNumInputsEqual(input_shapes.size());
//   return {input_shapes.at(0)};
// }

// Tensor MakeDropoutOp(Tensor input, double keep_prob, bool recompute,
//                      bool inplace, const OpMeta& op_meta);

// Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor output, double keep_prob,
//                              const OpMeta& op_meta);

// Tensor MakeDropoutGradientWithRecomputationOp(Tensor grad_output, OpId forward_op, double keep_prob,
//                                               const OpMeta& op_meta);


// } // namespace graph
// } // namespace hetu
