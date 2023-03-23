#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void range_kernel(spec_t start, spec_t step, size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = start + step * idx;
}

void ArangeCuda(double start, double step, NDArray& output, const Stream& stream) {

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "RangeCuda", [&]() {
      range_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<spec_t>(start), static_cast<spec_t>(step), size, output->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
