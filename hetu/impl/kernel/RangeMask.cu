#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void RangeMaskCuda(const NDArray& input, int64_t min, int64_t max,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RangeMaskCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                           return ((static_cast<int64_t>(x) >= min) &&
                                                   (static_cast<int64_t>(x) <= max)) ? 0 : 1;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
