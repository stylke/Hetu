#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void where_kernel(const spec_t* cond, const spec_t* arr1,
                             const spec_t* arr2, spec_t* output, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = cond[idx] ? arr1[idx] : arr2[idx];
}

void WhereCuda(const NDArray& cond, const NDArray& inputA,
               const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_EXCHANGABLE(cond, inputA);
  HT_ASSERT_EXCHANGABLE(cond, inputB);
  HT_ASSERT_EXCHANGABLE(cond, output);

  size_t size = cond->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    cond->dtype(), spec_t, "WhereCuda", [&]() {
      where_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        cond->data_ptr<spec_t>(), inputA->data_ptr<spec_t>(),
        inputB->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size);
    });
}

} // namespace impl
} // namespace hetu
