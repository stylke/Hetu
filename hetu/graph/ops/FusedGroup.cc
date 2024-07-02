// #include "hetu/graph/ops/FusedGroup.h"
// #include "hetu/graph/headers.h"
// #include "hetu/graph/ops/kernel_links.h"

// namespace hetu {
// namespace graph {


// FusedType OpType2FusedType(OpType optype){
//   if (optype == "AddByConstOp")
//     return FusedType::ADDCONST;
//   else if (optype == "SubByConstOp")
//     return FusedType::SUBCONST;
//   else if (optype == "SubFromConstOp")
//     return FusedType::SUBFROMCONST;
//   else if (optype == "MulByConstOp")
//     return FusedType::MULCONST;
//   else if (optype == "DivByConstOp")
//     return FusedType::DIVCONST;
//   else if (optype == "DivFromConstOp")
//     return FusedType::DIVFROMCONST;
//   else if (optype == "PowOp")
//     return FusedType::POW;
//   else if (optype == "ExpOp")
//     return FusedType::EXP;
//   else if (optype == "LogOp")
//     return FusedType::LOG;
//   else if (optype == "AbsOp")
//     return FusedType::ABS;
//   else if (optype == "SqrtOp")
//     return FusedType::SQRT;
//   else if (optype == "Fused_Output")
//     return FusedType::OUTPUT;
//   else
//     return FusedType::UNKNOWN;
// }

// void FusedGroupOpImpl::DoCompute(Operator& op,
//                                  const NDArrayList& inputs, NDArrayList& outputs,
//                                  RuntimeContext& ctx) const {
//   // use subgraph to gen compute plan.
//   SubGraph sub = subgraph();
//   FusedGroupParam fusedgroup;
//   fusedgroup.num_tensors = 0;
//   fusedgroup.num_inputs = sub.num_inputs;
//   fusedgroup.num_outputs = sub.num_outputs;
//   fusedgroup.param_info = sub.param_info;
//   for (int i = 0; i < sub.fetches.size(); ++i) {
//     fusedgroup.num_input_output.emplace_back(sub.input_idx[i].size());
//     for (int j = 0; j < sub.input_idx[i].size(); ++j) {
//       fusedgroup.input_output_idx.emplace_back(sub.input_idx[i][j]);
//     }
//     fusedgroup.num_input_output.emplace_back(sub.output_idx[i].size());
//     for (int j = 0; j < sub.output_idx[i].size(); ++j) {
//       fusedgroup.input_output_idx.emplace_back(sub.output_idx[i][j]);
//       if (sub.output_idx[i][j] + 1 > fusedgroup.num_tensors)
//         fusedgroup.num_tensors = sub.output_idx[i][j] + 1;
//     }
//   }
//   HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
//                                hetu::impl::FusedGroup, inputs, fusedgroup,
//                                outputs, op->instantiation_ctx().stream());
// }

// HTShapeList FusedGroupOpImpl::DoInferShape(Operator& op,
//                                            const HTShapeList& input_shapes,
//                                            RuntimeContext& ctx) const { 
//   HTShapeList output_shapes;
//   for (auto& output: op->outputs()) {
//     output_shapes.emplace_back(output->shape());
//   }
//   HT_LOG_INFO << "FUSEDSHAPE:" << output_shapes;
//   return output_shapes;
// }

// void FusedGroupOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
//                                       const OpMeta& op_meta) const {
//   HT_ASSERT(outputs.size() == _subgraph.ds_list.size());
//   for (int i = 0; i < outputs.size(); ++i) {
//     outputs[i]->set_distributed_states(_subgraph.ds_list[i]);
//   }
// }

// TensorList MakeFusedGroupOp(TensorList inputs, SubGraph subgraph, OpMeta op_meta) {
//   return Graph::MakeOp(
//           std::make_shared<FusedGroupOpImpl>(subgraph),
//           std::move(inputs),
//           std::move(op_meta))->outputs();
// }

// } // namespace graph
// } // namespace hetu
