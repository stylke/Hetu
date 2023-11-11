#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void dot_kernel(const spec_t* inputA, const spec_t* inputB,
                           size_t size, size_t size2, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = inputA[idx] * inputB[(int) (idx % size2)];
}

void MatDotCuda(const NDArray& inputA, const NDArray& inputB, NDArray& output,
                const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);
  HT_ASSERT_EXCHANGABLE(inputA, output);

  size_t size = inputA->numel();
  size_t size2 = inputB->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "MatDotCuda", [&]() {
      dot_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size, size2,
        output->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

} // namespace impl
} // namespace hetu
