#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__device__ spec_t sgn(spec_t x) {
    if (x == 0.0)
        return 0.0;
    return x / abs(x);
}

template <typename spec_t>
__global__ void norm_kernel(const spec_t* input, spec_t* output, size_t size, 
                            int64_t p, size_t before_dim_size, size_t reduce_dim_size, 
                            size_t after_dim_size) {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= size)
//     return;
  __shared__ spec_t shared_sum[32];

  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
      stride = after_dim_size * blockDim.x;
      start_ptr = x * reduce_dim_size * after_dim_size + y
                  + threadIdx.x * after_dim_size;
      end_ptr = x * reduce_dim_size * after_dim_size + y
                + reduce_dim_size * after_dim_size;
  } else {
      size_t cols_per_thread =
          (reduce_dim_size + blockDim.x - 1) / blockDim.x;
      size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y
                              + reduce_dim_size * after_dim_size;
      start_ptr = x * reduce_dim_size * after_dim_size + y
                  + threadIdx.x * cols_per_thread * after_dim_size;
      end_ptr =
          min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
      stride = after_dim_size;
  }
  size_t output_ptr = x * after_dim_size + y;
  if (start_ptr >= end_ptr)
      return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
      sum_thread += hetu::cuda::cuda_pow(abs(input[ptr]), spec_t(p));
  hetu::cuda::BlockReduceSum(sum_thread, shared_sum);
  if (threadIdx.x == 0)
      output[output_ptr] = hetu::cuda::cuda_pow(sum_thread, spec_t(1.0 / p));
}

template <typename spec_t>
__global__ void norm_gradient_kernel(const spec_t *input, const spec_t *norm,
                                     const spec_t *grad, spec_t *output, int64_t p,
                                     size_t reduce_dim_size,
                                     size_t after_dim_size, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int na = idx / (reduce_dim_size * after_dim_size);
  int nc = (idx % (reduce_dim_size * after_dim_size)) % after_dim_size;
  int idx_y = na * after_dim_size + nc;

  spec_t input_val = input[idx];
  spec_t grad_val = grad[idx_y];

  if (p == 1) {
      output[idx] = sgn(input_val) * grad_val;
  } else if (p == 2) {
      spec_t norm_val = norm[idx_y];
      if (norm_val == 0)
          output[idx] = 0;
      else
          output[idx] = grad_val * input_val / norm_val;
  } else if (p > 2) {
      spec_t norm_val = norm[idx_y];
      if (norm_val == 0)
          output[idx] = 0;
      else
          output[idx] = input_val * pow(abs(input_val), p - 2) * grad_val
                        / pow(norm_val, p - 1);
  }
}



void NormCuda(const NDArray& input, NDArray& output, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t before_dim_size, reduce_dim_size, after_dim_size;
  before_dim_size = reduce_dim_size = after_dim_size = 1;
  for (int i = 0; i < input->ndim(); ++i) {
      if (i < dim)
          before_dim_size *= input->shape(i);
      else if (i == dim)
          reduce_dim_size = input->shape(i);
      else
          after_dim_size *= input->shape(i);
  }

  size_t size = before_dim_size * after_dim_size;

  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = 32;
  blocks.x = before_dim_size * after_dim_size;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "NormCuda", [&]() {
      norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, p, 
        before_dim_size, reduce_dim_size, after_dim_size);
    });
        // CudaStreamSynchronize(cuda_stream);
    //   HT_LOG_INFO << output->data_ptr<void>();
}

void NormGradientCuda(const NDArray& input, const NDArray& output, const NDArray& output_grad,
                      NDArray& input_grad, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t reduce_dim_size, after_dim_size, size;
  reduce_dim_size = after_dim_size = size = 1;
  for (int i = 0; i < input->ndim(); ++i) {
      size *= input->shape(i);
      if (i == dim)
          reduce_dim_size = input->shape(i);
      else if (i > dim)
          after_dim_size *= input->shape(i);
  }

  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "NormGradientCuda", [&]() {
      norm_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), 
      input_grad->data_ptr<spec_t>(), p, reduce_dim_size, after_dim_size, size);
    });
}

} // namespace impl
} // namespace hetu
