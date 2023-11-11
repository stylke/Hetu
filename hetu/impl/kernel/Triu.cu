#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void triutril_kernel(const spec_t* input, spec_t* output, bool lower,
                                int64_t H, int64_t W, int64_t diagonal, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int row = (idx / W) % H;
  int col = idx % W;
  bool mask = lower ? (col - row > diagonal) : (col - row < diagonal);
  output[idx] = mask ? 0 : input[idx];
}

void TriuTrilCuda(const NDArray& input, NDArray& output, bool lower,
                  int64_t diagonal, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  int64_t ndim = input->ndim();
  int64_t H = input->shape(ndim - 2);
  int64_t W = input->shape(ndim - 1);
  if (diagonal < 0)
    diagonal += ndim;
  if (size == 0)
    return; 
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TriuTrilCuda", [&]() {
      triutril_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        lower, H, W, diagonal, size);
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
