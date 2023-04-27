#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void where_kernel(const int64_t* cond, const spec_t* arr1,
                             const spec_t* arr2, spec_t* output, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  bool cond_bit = bool(cond[idx]);
  output[idx] = cond_bit ? arr1[idx] : arr2[idx];
}

void WhereCuda(const NDArray& cond, const NDArray& inputA,
               const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_EXCHANGABLE(inputA, inputB);

  size_t size = cond->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "WhereCuda", [&]() {
      where_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        cond->data_ptr<int64_t>(), inputA->data_ptr<spec_t>(),
        inputB->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size);
    });
  // CudaStreamSynchronize(cuda_stream);
  // HT_LOG_INFO << cond << "\n" << inputA << "\n"
  // << inputB<< "\n" << output;
}

} // namespace impl
} // namespace hetu
