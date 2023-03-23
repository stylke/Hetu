// #pragma once

// #include "hetu/graph/operator.h"
// #include "hetu/graph/utils/tensor_utils.h"

// namespace hetu {
// namespace graph {

// using OpDim = std::vector<std::string>;
// using OpDimList = std::vector<OpDim>;
// using LabelMap = std::unordered_map<std::string, int>;

// class EinsumOpImpl;
// class EinsumOp;
// class EinsumGradientOpImpl;
// class EinsumGradientOp;

// class EinsumOpImpl : public OpInterface {
//  public:
//   EinsumOpImpl(const std::string& msg)
//   : OpInterface(quote(EinsumOp)),
//     _msg(msg),
//     input_dims(),
//     output_dims() {
//   }

//   inline std::string fetch_msg() const {
//     return _msg;
//   }

//   inline HTShape get_grad_shape() const {
//     return _grad_shape;
//   }

//   void set_grad_shape(HTShape shape) {
//     _grad_shape = shape;
//   }

//  protected:
//   void ParseMsg();

//   std::vector<NDArrayMeta>
//   DoInferMeta(const TensorList& inputs) const override {
    
//     return {output_meta};
//   }

//   TensorList DoGradient(Operator& op,
//                         const TensorList& grad_outputs) const override;

//   HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
//                            RuntimeContext& runtime_ctx) const override;

//   void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
//                  RuntimeContext& runtime_ctx) const override;

//   std::string _msg;

//   std::vector<std::string> _input_msgs;

//   std::string _output_msg;

//   OpDimList input_dims;

//   OpDimList output_dims;

//   LabelMap num_labels;

//   LabelMap output_labels_idx;

//   int output_size;

//   int num_output_labels;

//   int elli_len;

//   std::vector<int> input_elli_len;

//   int elli_pos;

//   HTShape _grad_shape;
// };

// class EinsumOp final : public OpWrapper<EinsumOpImpl> {
//  public:
//   EinsumOp(const std::string& msg, const TensorList& inputs,
//            const OpMeta& op_meta = OpMeta())
//   : OpWrapper<EinsumOpImpl>(make_ptr<EinsumOpImpl>(
//       EinsumOpImpl::constrcutor_access_key(), msg, inputs, op_meta)) {}
// };

// class EinsumGradientOpImpl : public OpInterface {
//  private:
//   friend class EinsumGradientOp;
//   struct constrcutor_access_key {};

//  public:
//   EinsumGradientOpImpl(const constrcutor_access_key&, const std::string& msg,
//                       const TensorList& inputs, Tensor ori_output,
//                       Tensor ori_input, const OpMeta& op_meta = OpMeta())
//   : OpInterface(quote(EinsumGradientOp), inputs, op_meta),
//     pred(ori_output),
//     pred_in(ori_input),
//     _msg(msg),
//     input_dims(),
//     output_dims() {
//     // HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
//     // AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()));
//     DoInferMeta();
//   }

//   inline std::string fetch_msg() const {
//     return _msg;
//   }

//   Tensor pred;

//   Tensor pred_in;

//  protected:
//   void ParseMsg(const HTShapeList& input_shapes);

//   void DoInferMeta() override;

//   void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
//                  RuntimeContext& ctx) override;

//   HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

//   std::string _msg;

//   std::vector<std::string> _input_msgs;

//   std::string _output_msg;

//   OpDimList input_dims;

//   OpDimList output_dims;

//   LabelMap undefined_labels;

//   LabelMap num_labels;

//   LabelMap output_labels_idx;

//   int output_size;

//   int ori_output_size;

//   int num_output_labels;

//   int elli_len;

//   std::vector<int> input_elli_len;

//   int elli_pos;
// };

// class EinsumGradientOp final : public OpWrapper<EinsumGradientOpImpl> {
//  public:
//   EinsumGradientOp(const std::string& msg, const TensorList& inputs,
//                    Tensor ori_output, Tensor ori_input,
//                    const OpMeta& op_meta = OpMeta())
//   : OpWrapper<EinsumGradientOpImpl>(make_ptr<EinsumGradientOpImpl>(
//       EinsumGradientOpImpl::constrcutor_access_key(), msg, inputs, ori_output,
//       ori_input, op_meta)) {}
// };

// } // namespace autograd
// } // namespace hetu
