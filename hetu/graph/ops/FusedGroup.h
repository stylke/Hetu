// #pragma once

// #include "hetu/graph/operator.h"
// #include "hetu/graph/utils/tensor_utils.h"
// #include "hetu/graph/graph.h"

// namespace hetu {
// namespace graph {

// class FusedGroupOpImpl;
// class FusedGroupOp;


// FusedType OpType2FusedType(OpType optype);

// class FusedGroupOpImpl final : public OpInterface {
//  public:
//   FusedGroupOpImpl(SubGraph subgraph, OpMeta op_meta = OpMeta())
//   : OpInterface(quote(FusedGroupOp)), _subgraph(subgraph) {
//   }

//   inline uint64_t op_indicator() const noexcept override {
//     return FUSED_GROUP_OP;
//   }

//   SubGraph subgraph() const {
//     return _subgraph;
//   }

//  protected:
//   std::vector<NDArrayMeta>
//   DoInferMeta(const TensorList& inputs) const override {
//     return _subgraph.meta_list;
//   }

//   void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
//                       const OpMeta& op_meta) const override;

//   HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
//                            RuntimeContext& runtime_ctx) const override;

//   void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
//                  RuntimeContext& runtime_ctx) const override;

//   SubGraph _subgraph;


//  public:
//   inline bool require_contig_inputs() const override {
//     return false;
//   }

//   bool operator==(const OpInterface& rhs) const override {
//     if (OpInterface::operator==(rhs)) {
//       const auto& rhs_ = reinterpret_cast<const FusedGroupOpImpl&>(rhs);
//       return (subgraph().topo == rhs_.subgraph().topo &&
//               subgraph().fetches == rhs_.subgraph().fetches);
//     }
//     return false;
//   }
// };

// TensorList MakeFusedGroupOp(TensorList inputs, SubGraph subgraph, OpMeta op_meta = OpMeta());

// } // namespace graph
// } // namespace hetu
