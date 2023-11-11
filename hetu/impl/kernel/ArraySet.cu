#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void array_set_kernel(spec_t* arr, spec_t value, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    arr[idx] = value;
}

void ArraySetCuda(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "ArraySetCuda", [&]() {
      array_set_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        data->data_ptr<spec_t>(), static_cast<spec_t>(value), size);
    });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
