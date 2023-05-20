#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

std::pair<HTAxes, HTKeepDims> GradInfer(const HTShapeList& input_shapes) {
  HTShape output_shape = input_shapes[3];
  HTShape input_shape = input_shapes[2];
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim);
  size_t diff = ndim - input_shape.size();
  HTAxes add_axes(diff);
  HTKeepDims keep_dims(diff);
  size_t len = diff + input_shape.size();
  HTShape n_input_shape(len);
  for (size_t i = 0; i < diff; ++i) {
    add_axes[i] = i;
    keep_dims[i] = false;
    n_input_shape[i] = 1;
  }
  for (size_t i = diff; i < len; ++i) {
    n_input_shape[i] = input_shape[i - diff];
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (output_shape[i] == -1) {
      output_shape[i] = n_input_shape[i];
    }
    // HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
    if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
      add_axes.emplace_back(i);
      keep_dims.emplace_back(true);
    }
  }
  return std::make_pair(add_axes, keep_dims);
}

void AddElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::add(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AddElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeAddElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeAddElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList AddElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void AddElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "AddElewiseOpDef: distributed states for input a and input b must be valid!";
  // allow sum/sub input tensor states to be partial
  // HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
  //   << "Tensor a & b shouldn't be partial";
  // local shape
  HTShape shape_a = inputs.at(0)->shape();
  HTShape shape_b = inputs.at(1)->shape();

  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    outputs.at(0)->set_distributed_states(ds_a);
  } else {
    outputs.at(0)->set_distributed_states(ds_b);
  }
}

void AddByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::add(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AddByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? grad_outputs.at(0) : Tensor()};
}

HTShapeList AddByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void SubElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::sub(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeSubElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeSubElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList SubElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void SubElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "SubElewiseOpDef: distributed states for input a and input b must be valid!";
  // HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
  //   << "Tensor a & b shouldn't be partial";  
  // local shape
  HTShape shape_a = inputs.at(0)->shape();
  HTShape shape_b = inputs.at(1)->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    outputs.at(0)->set_distributed_states(ds_a);
  } else {
    outputs.at(0)->set_distributed_states(ds_b);
  }
}

void SubByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::sub(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? grad_outputs.at(0) : Tensor()};
}

HTShapeList SubByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}


void SubFromConstOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  NDArray::sub(const_value(), inputs.at(0),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubFromConstOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto grad_input =  op->requires_grad(0) ? MakeNegateOp(grad_outputs.at(0), 
                                           op->grad_op_meta().set_name(op->grad_name()))
                                         : Tensor();
  return {grad_input};
}

HTShapeList SubFromConstOpImpl::DoInferShape(Operator& op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void NegateOpImpl::DoCompute(Operator& op,
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  NDArray::neg(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList NegateOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeNegateOp(grad_outputs.at(0), 
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList NegateOpImpl::DoInferShape(Operator& op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void MulElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::mul(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList MulElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeMulElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeMulElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList MulElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void MulElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "MulElewiseOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";  
  // local shape
  HTShape shape_a = inputs.at(0)->shape();
  HTShape shape_b = inputs.at(1)->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    outputs.at(0)->set_distributed_states(ds_a);
  } else {
    outputs.at(0)->set_distributed_states(ds_b);
  }
}

void MulByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::mul(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList MulByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeMulByConstOp(grad_outputs.at(0), const_value(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList MulByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}


void DivElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::div(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeDivElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeDivElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList DivElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void DivElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "DivElewiseOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";   
  // local shape
  HTShape shape_a = inputs.at(0)->shape();
  HTShape shape_b = inputs.at(1)->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    outputs.at(0)->set_distributed_states(ds_a);
  } else {
    outputs.at(0)->set_distributed_states(ds_b);
  }
}

void DivByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::div(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDivByConstOp(grad_outputs.at(0), const_value(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList DivByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void DivFromConstOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  NDArray::div(const_value(), inputs.at(0),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivFromConstOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeMulElewiseOp(MakeNegateOp(MakeDivElewiseOp(
                                          op->output(0), op->input(1), g_op_meta),
                                          g_op_meta), grad_outputs.at(0),
                                          g_op_meta.set_name(op->grad_name(1)))
                                        : Tensor();
  return {grad_input};
}

HTShapeList DivFromConstOpImpl::DoInferShape(Operator& op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ReciprocalOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::reciprocal(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ReciprocalOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  // 1 / (x^2) = (1 / x) * (1 / x)
  if (!op->requires_grad(0))
    return {Tensor()};
  auto ret = MakeMulElewiseOp(op->output(0), op->output(0), g_op_meta);
  ret = MakeNegateOp(ret, g_op_meta);
  ret = MakeMulElewiseOp(ret, grad_outputs.at(0), g_op_meta.set_name(op->grad_name()));
  return {ret};
}

HTShapeList ReciprocalOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void AddElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::copy(inputs.at(0), op->instantiation_ctx().stream_index, unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList AddElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void AddElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta) const {
  // ds for grad_input is equal to input; TODO: special case: input is partial
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

void SubElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0)
    NDArray::copy(inputs.at(0), op->instantiation_ctx().stream_index, unreduced);
  else 
    NDArray::neg(inputs.at(0), op->instantiation_ctx().stream_index, unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList SubElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void SubElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta) const {
  // ds for grad_input is equal to input; TODO: special case: input is partial
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());  
}

void MulElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::mul(inputs.at(0), inputs.at(1), op->instantiation_ctx().stream_index, unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList MulElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void MulElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

void DivElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0) {
    NDArray::div(inputs.at(0), inputs.at(1), op->instantiation_ctx().stream_index, unreduced);
  }
  else {
    NDArray::mul(inputs.at(0), inputs.at(3), op->instantiation_ctx().stream_index, unreduced);
    NDArray::div(unreduced, inputs.at(2), op->instantiation_ctx().stream_index, unreduced);
    NDArray::neg(unreduced, op->instantiation_ctx().stream_index, unreduced); 
  }
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList DivElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void DivElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

Tensor MakeAddElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddElewiseOpImpl>(),
           {std::move(a), std::move(b)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubElewiseOpImpl>(),
           {std::move(a), std::move(b)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulElewiseOpImpl>(),
           {std::move(a), std::move(b)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivElewiseOpImpl>(),
           {std::move(a), std::move(b)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubFromConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubFromConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivByConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivFromConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivFromConstOpImpl>(value),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({a->shape(), b->shape(), input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<AddElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b), std::move(input), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({a->shape(), b->shape(), input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<SubElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b), std::move(input), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({a->shape(), b->shape(), input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<MulElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b), std::move(input), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({a->shape(), b->shape(), input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<DivElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b), std::move(input), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeNegateOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<NegateOpImpl>(),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeReciprocalOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<ReciprocalOpImpl>(),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
