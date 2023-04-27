#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void eye_kernel(spec_t* output, size_t size, size_t ncols) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t row = idx / ncols;
  size_t col = idx % ncols;
  if (row == col)
    output[idx] = 1;
  else
    output[idx] = 0;
}

void EyeCuda(NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT(output->ndim() == 2);

  size_t size = output->numel();
  size_t ncols = output->shape(1);
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "EyeCuda", [&]() {
      eye_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), size, ncols);
    });
}

} // namespace impl
} // namespace hetu
