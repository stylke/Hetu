#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void gather_cpu(const spec_t* input, const spec_t* ids,
                size_t size, size_t before_stride,
                size_t after_stride, size_t cur_stride,
                spec_t* output) {
  
  for (size_t idx = 0; idx < size; ++idx) {
    size_t b_index = idx / (cur_stride * after_stride);
    size_t p_index = idx % (cur_stride * after_stride);
    size_t a_index = p_index % after_stride;
    size_t id_num = int(ids[idx]);
    size_t i_index =
      b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
    output[idx] = input[i_index];
  }
}

template <typename spec_t>
void array_zero_set_cpu(spec_t* input, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) 
    input[idx] = 0;
}

template <typename spec_t>
void gather_gradient_cpu(const spec_t* grad_output, const spec_t* ids,
                                       size_t size, size_t before_stride,
                                       size_t after_stride, size_t cur_stride,
                                       spec_t* grad_input) {
  for (int64_t idx = 0; idx < size; ++idx) {
    size_t b_index = idx / (cur_stride * after_stride);
    size_t p_index = idx % (cur_stride * after_stride);
    size_t c_index = p_index / after_stride;
    size_t a_index = p_index % after_stride;
    size_t id_num = int(ids[idx]);
    size_t i_index =
      b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
    grad_input[i_index] += grad_output[idx];
  }
}

void GatherCpu(const NDArray& input, const NDArray& id, NDArray& output,
               size_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(id->ndim() == 1)
    << "invalid index shape.Expect dim=1, but get" << id->ndim();
  size_t before_stride = 1;
  size_t after_stride = 1;
  size_t cur_stride = output->shape(dim);
  HT_ASSERT(id->numel() == cur_stride && id->shape() == output->shape())
    << "Invalid shapes.Index shape:" << id->shape()
    << "Input shape:" << input->shape() << "Output shape:" << output->shape();

  for (size_t i = 0; i < dim; ++i) {
    before_stride *= output->shape(i);
  }
  for (size_t i = dim + 1; i < input->ndim(); ++i) {
    after_stride *= output->shape(i);
  }
  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GatherCuda", [&]() {
      gather_cpu<spec_t>(
        input->data_ptr<spec_t>(), id->data_ptr<spec_t>(), size, before_stride,
        after_stride, cur_stride, output->data_ptr<spec_t>());
    });
}

void GatherGradientCpu(const NDArray& grad_output, const NDArray& id, NDArray& grad_input,
                       size_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(grad_output);
  HT_ASSERT_SAME_DEVICE(grad_output, id);
  HT_ASSERT_SAME_DEVICE(grad_output, grad_input);
  HT_ASSERT(id->ndim() == 1)
    << "invalid index shape.Expect dim=1, but get" << id->ndim();
  size_t before_stride = 1;
  size_t after_stride = 1;
  size_t cur_stride = grad_input->shape(dim);
  for (size_t i = 0; i < dim; ++i) {
    before_stride *= grad_output->shape(i);
  }
  for (size_t i = dim + 1; i < grad_input->ndim(); ++i) {
    after_stride *= grad_output->shape(i);
  }
  size_t size = grad_output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    grad_output->dtype(), spec_t, "ArraySetZeroCpu", [&]() {
      array_zero_set_cpu<spec_t>(
        grad_input->data_ptr<spec_t>(), grad_input->numel());
    });
  HT_DISPATCH_FLOATING_TYPES(
    grad_output->dtype(), spec_t, "GatherGradientCpu", [&]() {
      gather_cpu<spec_t>(
        grad_output->data_ptr<spec_t>(), id->data_ptr<spec_t>(), size, before_stride,
        after_stride, cur_stride, grad_input->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
