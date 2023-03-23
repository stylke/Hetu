#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSumExp(spec_t& val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSumExp(spec_t& val,
                                                  spec_t* shared) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSumExp(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

  if (wid == 0) {
    val = WarpReduceSumExp(val);
    shared[0] = val;
  }
}

template <typename spec_t>
__global__ void softmax_kernel(const spec_t* input, spec_t* output,
                               size_t befor_dim_size,
                               size_t reduce_dim_size,
                               size_t after_dim_size) {
  __shared__ spec_t shared_sum[32];

  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
    stride = after_dim_size * blockDim.x;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = (reduce_dim_size + blockDim.x - 1) / blockDim.x;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y +
      threadIdx.x * cols_per_thread * after_dim_size;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
  }
  if (start_ptr >= end_ptr)
    return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
    sum_thread += hetu::cuda::cuda_exp(input[ptr]);

  BlockReduceSumExp(sum_thread, shared_sum);
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) 
    output[ptr] = hetu::cuda::cuda_exp(input[ptr]) / shared_sum[0];
}

void SoftmaxCuda(const NDArray& input, NDArray& output, int64_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t before_dim_size = 1, reduce_dim_size, after_dim_size = 1;
  reduce_dim_size = input->shape(dim);
  for (size_t i = 0; i < input->ndim(); ++i) {
    if (i < dim)
      before_dim_size *= input->shape(i);
    else if (i > dim)
      after_dim_size *= input->shape(i);
  }

  int blocks = before_dim_size * after_dim_size;
  int threads = hetu::impl::GetThreadNum(reduce_dim_size);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftMaxCuda", [&]() {
      softmax_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
        before_dim_size, reduce_dim_size, after_dim_size);
    });
}

template <typename spec_t>
__global__ void softmax_grad_kernel(const spec_t* output, const spec_t* output_grad,
                                    spec_t* input_grad,
                                    size_t befor_dim_size,
                                    size_t reduce_dim_size,
                                    size_t after_dim_size) {
  __shared__ spec_t shared_sum[32];

  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
    stride = after_dim_size * blockDim.x;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = (reduce_dim_size + blockDim.x - 1) / blockDim.x;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y +
      threadIdx.x * cols_per_thread * after_dim_size;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
  }
  if (start_ptr >= end_ptr)
    return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
    sum_thread += output_grad[ptr] * output[ptr];

  BlockReduceSumExp(sum_thread, shared_sum);
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride) 
    input_grad[ptr] = output_grad[ptr] * output[ptr] - output[ptr] * shared_sum[0];
}

void SoftmaxGradientCuda(const NDArray& input_Y, const NDArray& output_grad,
                         NDArray& input_grad, int64_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_Y);
  HT_ASSERT_SAME_DEVICE(input_Y, output_grad);
  HT_ASSERT_SAME_DEVICE(input_Y, input_grad);

  size_t before_dim_size = 1, reduce_dim_size, after_dim_size = 1;
  reduce_dim_size = input_Y->shape(dim);
  for (size_t i = 0; i < input_Y->ndim(); ++i) {
    if (i < dim)
      before_dim_size *= input_Y->shape(i);
    else if (i > dim)
      after_dim_size *= input_Y->shape(i);
  }

  int blocks = before_dim_size * after_dim_size;
  int threads = hetu::impl::GetThreadNum(reduce_dim_size);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    input_Y->dtype(), spec_t, "SoftMaxCuda", [&]() {
      softmax_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input_Y->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(),
        input_grad->data_ptr<spec_t>(),
        before_dim_size, reduce_dim_size, after_dim_size);
    });
}

} // namespace impl
} // namespace hetu
