#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void clamp_kernel(const spec_t* input, const spec_t min_val,
                             const spec_t max_val, size_t size, spec_t* output,
                             const OffsetCalculator* in_offset_calculator,
                             const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  spec_t min_v = min_val, max_v = max_val;
  auto in_offset = in_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  spec_t cur_v = input[in_offset];
  if (cur_v < min_v)
    output[out_offset] = min_v;
  else if (cur_v > max_v) {
    output[out_offset] = max_v;
  }
  else 
    output[out_offset] = cur_v;
}

template <typename spec_t>
__global__ void clamp_elewise_kernel(const spec_t* input, const spec_t* min_val,
                                     const spec_t* max_val, size_t size, spec_t* output,
                                     const OffsetCalculator* in_offset_calculator,
                                     const OffsetCalculator* min_offset_calculator,
                                     const OffsetCalculator* max_offset_calculator,
                                     const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_offset = in_offset_calculator->get(idx);
  auto min_offset = min_offset_calculator->get(idx);
  auto max_offset = max_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  spec_t min_v = min_val[min_offset], max_v = max_val[max_offset];
  spec_t cur_v = input[in_offset];
  if (cur_v < min_v)
    output[out_offset] = min_v;
  else if (cur_v > max_v) {
    output[out_offset] = max_v;
  }
  else 
    output[out_offset] = cur_v;
}

void ClampCuda(const NDArray& input, double min_val, double max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      clamp_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), min_val, max_val, size, output->data_ptr<spec_t>(),
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

void ClampElewiseCuda(const NDArray& input, const NDArray& min_val, const NDArray& max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, min_offset_calculator_arr,
          max_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *min_offset_calculator,
                   *max_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(min_offset_calculator_arr, min_offset_calculator) =
    AllocOffsetCalculator(min_val, stream);
  std::tie(max_offset_calculator_arr, max_offset_calculator) =
    AllocOffsetCalculator(max_val, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) =
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      clamp_elewise_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), min_val->data_ptr<spec_t>(), max_val->data_ptr<spec_t>(), 
        size, output->data_ptr<spec_t>(), in_offset_calculator, min_offset_calculator,
        max_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, min_val, max_val, output, in_offset_calculator_arr,
                      min_offset_calculator_arr, max_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
