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
  loss[idx] = -(pred[curid] - maxval) + spec_t(hetu::cuda::cuda_log(sum));
}

template <typename spec_t>
__global__ void
softmax_cross_entropy_sparse_gradient_kernel(const spec_t* pred, const int64_t* label,
                                             const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                                             const int64_t ignored_index,
                                             spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;

  if(int64_t(label[idx]) == ignored_index) {
    for (size_t i = 0; i < n_cols; ++i) {
        size_t curid = idx * n_cols + i;
        output[curid] = 0;
    }
    return;        
  }
  
  spec_t maxval = pred[idx * n_cols];

  for (size_t i = 1; i < n_cols; ++i) {
      maxval = MAX(maxval, pred[idx * n_cols + i]);
  }

  float sum = 0;
  for (size_t i = 0; i < n_cols; ++i) {
      sum += hetu::cuda::cuda_exp(float(pred[idx * n_cols + i] - maxval));
  }
  for (size_t i = 0; i < n_cols; ++i) {
      size_t curid = idx * n_cols + i;
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
  threads.x = MIN(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseCuda", [&]() {
      softmax_cross_entropy_sparse_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        ignored_index, loss->data_ptr<spec_t>());
    });
  // CudaStreamSynchronize(cuda_stream);
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
