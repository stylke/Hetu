#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void concatenate_kernel(const spec_t* input, spec_t* output,
                                   int input_width, int output_width,
                                   int offset, int concat_size, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int prev_ind = idx / concat_size;
  int mid_ind = prev_ind % input_width + offset;
  prev_ind = prev_ind / input_width;
  int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
  output[out_ind] = input[idx];
}

template <typename spec_t>
__global__ void concatenate_gradient_kernel(const spec_t* output_grad,
                                            spec_t* input_grad, int input_width,
                                            int output_width, int offset,
                                            int concat_size, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int prev_ind = idx / concat_size;
  int mid_ind = prev_ind % input_width + offset;
  prev_ind = prev_ind / input_width;
  int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
  input_grad[idx] = output_grad[out_ind];
}

void ConcatenateCuda(const NDArray& input, NDArray& output, size_t axis,
                     size_t offset, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = input->numel();
  int now_ndim = output->ndim();
  HT_ASSERT(input->ndim() == now_ndim);
  int num_concats = 1;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    concat_size *= cur_dim;
  }
  int input_width = input->shape(axis);
  int output_width = output->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ConcatenateCuda", [&]() {
      concatenate_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_width,
        output_width, offset, concat_size, size);
    });
}

void ConcatenateGradientCuda(const NDArray& output_grad, NDArray& input_grad,
                             size_t axis, size_t offset, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  int now_ndim = output_grad->ndim();
  HT_ASSERT(now_ndim == input_grad->ndim());
  int num_concats = 1;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    concat_size *= cur_dim;
  }
  int output_width = output_grad->shape(axis);
  int input_width = input_grad->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatenateGradientCuda", [&]() {
      concatenate_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        input_width, output_width, offset, concat_size, size);
    });
}

} // namespace impl
} // namespace hetu
