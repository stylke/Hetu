#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class GroupOpImpl final : public OpInterface {
 public:
  GroupOpImpl() : OpInterface(quote(GroupOp)) {}

  uint64_t op_indicator() const noexcept {
    return GROUP_OP;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList&) const override {
    return {};
  }

  void DoCompute(Operator&, const NDArrayList&, NDArrayList&,
                 RuntimeContext&) const {}
};

Tensor MakeGroupOp(OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
