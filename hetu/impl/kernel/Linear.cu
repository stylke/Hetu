#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void bias_set_kernel(const spec_t* input, spec_t* output,
                                size_t input_size, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = input[idx % input_size];
}

void LinearCuda(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
                const NDArray& bias, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  size_t size = output->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());  
  
  bool bias_exist = bias.is_defined();
  if (bias_exist) {
    HT_ASSERT_SAME_DEVICE(a, bias);
    HT_ASSERT_NDIM(bias, 1);
    HT_ASSERT_SAME_DTYPE(a, bias);
    size_t input_size = bias->numel();
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      bias->dtype(), spec_t, "BiasSetCuda", [&]() {
        bias_set_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          bias->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_size, size);
      });
  }

  int32_t m = output->shape(1);
  int32_t n = output->shape(0);
  int32_t k = trans_a ? a->shape(0) : a->shape(1);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatMul", [&]() {
    spec_t alpha = 1;
    spec_t beta = bias_exist ? 1 : 0;
    cublas_gemm<spec_t>(cublas_handle, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                        trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
                        b->data_ptr<spec_t>(), trans_b ? k : m,
                        a->data_ptr<spec_t>(), trans_a ? n : k, &beta,
                        output->data_ptr<spec_t>(), m);
  });
}

} // namespace impl
} // namespace hetu
