#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void rangemask_kernel(const spec_t* input, int64_t min, 
                                 int64_t max, int64_t* output, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = (static_cast<int64_t>(input[idx]) >= min) && (static_cast<int64_t>(input[idx]) <= max) ? 0 : 1;
}

void RangeMaskCuda(const NDArray& input, int64_t min, int64_t max,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RangeMaskCuda", [&]() {
      rangemask_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), min, max, output->data_ptr<int64_t>(), size);
    });
}

} // namespace impl
} // namespace hetu
