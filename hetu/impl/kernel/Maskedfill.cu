#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void maskedfill_kernel(const spec_t* input, const int64_t* mask, 
                                  spec_t val, spec_t* output, size_t size,
                                  const OffsetCalculator* in_offset_calculator,
                                  const OffsetCalculator* mask_offset_calculator,
                                  const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto mask_offset = mask_offset_calculator->get(idx);
  bool mask_bit = bool(mask[mask_offset]);
  auto in_offset = in_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = mask_bit ? val : input[in_offset];
}

void MaskedfillCuda(const NDArray& input, const NDArray& mask,
                  double val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, mask_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *mask_offset_calculator,
                   *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(mask_offset_calculator_arr, mask_offset_calculator) = 
    AllocOffsetCalculator(mask, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "MaskfillCuda", [&]() {
      maskedfill_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), mask->data_ptr<int64_t>(),
        static_cast<spec_t>(val), output->data_ptr<spec_t>(), size,
        in_offset_calculator, mask_offset_calculator,
        out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, mask, output, in_offset_calculator_arr,
                      mask_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
