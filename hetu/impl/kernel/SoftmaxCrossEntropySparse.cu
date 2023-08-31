#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void softmax_cross_entropy_sparse_kernel(const spec_t* pred,
                                                    const int64_t* label, 
                                                    size_t n_rows, size_t n_cols,
                                                    const int64_t ignored_index,
                                                    spec_t* loss) {
  auto idx = blockIdx.x;
  size_t start_idx = idx * n_cols + threadIdx.x;
  size_t end_idx = (idx + 1) * n_cols;
  size_t stride = blockDim.x;
  if (start_idx >= end_idx)
    return;
  if (idx >= n_rows)
    return;
  if(int64_t(label[idx]) == ignored_index) {
    loss[idx] = 0;
    return;
  }  
  __shared__ spec_t buffer[32];
  __shared__ float buffer_f[32];
  __shared__ spec_t wrap_max[1];
  spec_t maxval = pred[start_idx];
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    maxval = hetu::cuda::cuda_max(pred[ptr], maxval);
  }
  hetu::cuda::BlockReduceArgmax(maxval, buffer, wrap_max);
  maxval = wrap_max[0];
  float sum = 0;
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    sum += hetu::cuda::cuda_exp(float(pred[ptr] - maxval));
  }
  hetu::cuda::BlockReduceSum(sum, buffer_f);
  if (threadIdx.x == 0) {
    size_t curid = idx * n_cols + int64_t(label[idx]);
    loss[idx] = - pred[curid] + maxval + spec_t(hetu::cuda::cuda_log(sum));
  }
}

template <typename spec_t>
__global__ void softmax_cross_entropy_sparse_kernel2(const spec_t* pred,
                                                    const int64_t* label, 
                                                    size_t n_rows, size_t n_cols,
                                                    const int64_t ignored_index,
                                                    spec_t* loss) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  if(int64_t(label[idx]) == ignored_index) {
    loss[idx] = 0;
    return;
  }  
  spec_t maxval = pred[idx * n_cols];
  for (size_t i = 1; i < n_cols; ++i) {
    maxval = hetu::cuda::cuda_max(maxval, pred[idx * n_cols + i]);
  }

  float sum = 0;
  for (int i = 0; i < n_cols; ++i) {
    sum += hetu::cuda::cuda_exp(float(pred[idx * n_cols + i] - maxval));
  }

  size_t curid = idx * n_cols + int64_t(label[idx]);
  // loss[idx] = -(pred[curid] - maxval) + hetu::cuda::cuda_log(sum);
  loss[idx] = - pred[curid] + maxval + spec_t(hetu::cuda::cuda_log(sum));
}

template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSum(spec_t val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <>
__forceinline__ __device__ bfloat16 WarpReduceSum<bfloat16>(bfloat16 val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  #if(__CUDA_ARCH__ >= 800)
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  #else
  float val_f = float(val);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val_f += __shfl_down_sync(mask, val_f, k, warpSize); 
  val = bfloat16(val_f); 
  #endif
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared, spec_t* warp_sum) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

  if (wid == 0) {
    val = WarpReduceSum(val);
    if (threadIdx.x == 0)
      warp_sum[0] = val;
  }
  __syncthreads();
}

template <typename spec_t>
__global__ void
softmax_cross_entropy_sparse_gradient_kernel(const spec_t* pred, const int64_t* label,
                                             const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                                             const int64_t ignored_index,
                                             spec_t* output) {
  auto idx = blockIdx.x;
  if (idx >= n_rows)
    return;
  size_t start_idx = idx * n_cols + threadIdx.x;
  size_t end_idx = (idx + 1) * n_cols;
  size_t stride = blockDim.x;
  if (start_idx >= end_idx)
    return;

  __shared__ spec_t buffer[32];
  __shared__ float buffer_f[32];
  __shared__ spec_t wrap_max[1];
  __shared__ float wrap_sum[1];
  spec_t maxval = pred[start_idx];
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    maxval = hetu::cuda::cuda_max(pred[ptr], maxval);
  }
  hetu::cuda::BlockReduceArgmax(maxval, buffer, wrap_max);
  maxval = wrap_max[0];
  float sum = 0;
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    sum += hetu::cuda::cuda_exp(float(pred[ptr] - maxval));
  }
  BlockReduceSum(sum, buffer_f, wrap_sum);
  sum = wrap_sum[0];
  for (size_t curid = start_idx; curid < end_idx; curid += stride) {
      size_t i = curid - idx * n_cols;
      if(i == int64_t(label[idx]))
        output[curid] = (hetu::cuda::cuda_exp(pred[curid] - maxval) / spec_t(sum) - 1.0) * grad_loss[idx];
      else
        output[curid] = (hetu::cuda::cuda_exp(pred[curid] - maxval) / spec_t(sum)) * grad_loss[idx];
  }
}

void SoftmaxCrossEntropySparseCuda(const NDArray& pred, const NDArray& label,
                                   NDArray& loss, const int64_t ignored_index, 
                                   const Stream& stream) {
  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(MAX(32, n_cols), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = n_rows;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseCuda", [&]() {
      softmax_cross_entropy_sparse_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        ignored_index, loss->data_ptr<spec_t>());
    });
}

void SoftmaxCrossEntropySparseGradientCuda(const NDArray& pred, const NDArray& label,
                                           const NDArray& grad_loss, NDArray& output,
                                           const int64_t ignored_index,
                                           const Stream& stream) {

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, 1024);
  blocks.x = DIVUP(n_rows, 1024);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(MAX(32, n_cols), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = n_rows;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseGradientCuda", [&]() {
      softmax_cross_entropy_sparse_gradient_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, n_cols,
        ignored_index, output->data_ptr<spec_t>());
    });  
}

} // namespace impl
} // namespace hetu
