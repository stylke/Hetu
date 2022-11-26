#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void concatenate_cpu(const spec_t* input, spec_t* output, int input_width,
                     int output_width, int offset, int concat_size,
                     size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    output[out_ind] = input[idx];
  }
}

template <typename spec_t>
void concatenate_gradient_cpu(const spec_t* output_grad, spec_t* input_grad,
                              int input_width, int output_width, int offset,
                              int concat_size, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    input_grad[idx] = output_grad[out_ind];
  }
}

void ConcatenateCpu(const NDArray& input, NDArray& output, size_t axis,
                    size_t offset, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t now_ndim = output->ndim();
  HT_ASSERT(input->ndim() == now_ndim);
  int num_concats = 1;
  for (size_t i = 0; i < axis; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    concat_size *= cur_dim;
  }
  int input_width = input->shape(axis);
  int output_width = output->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ConcatenateCpu", [&]() {
      concatenate_cpu<spec_t>(input->data_ptr<spec_t>(),
                              output->data_ptr<spec_t>(), input_width,
                              output_width, offset, concat_size, size);
    });
}

void ConcatenateGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                            size_t axis, size_t offset, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  size_t now_ndim = output_grad->ndim();
  HT_ASSERT(now_ndim == input_grad->ndim());
  int num_concats = 1;
  for (size_t i = 0; i < axis; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    concat_size *= cur_dim;
  }
  int output_width = output_grad->shape(axis);
  int input_width = input_grad->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatenateGradientCpu", [&]() {
      concatenate_gradient_cpu<spec_t>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        input_width, output_width, offset, concat_size, size);
    });
}

} // namespace impl
} // namespace hetu
