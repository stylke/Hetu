#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void where_kernel(const int64_t* cond, const spec_t* arr1,
                             const spec_t* arr2, spec_t* output, size_t size,
                             const OffsetCalculator* arr1_offset_calculator,
                             const OffsetCalculator* arr2_offset_calculator,
                             const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  bool cond_bit = bool(cond[idx]);
  auto arr1_offset = arr1_offset_calculator->get(idx);
  auto arr2_offset = arr2_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = cond_bit ? arr1[arr1_offset] : arr2[arr2_offset];
}

void WhereCuda(const NDArray& cond, const NDArray& inputA,
               const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_SAME_SHAPE(inputA, inputB);

  size_t size = cond->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray A_offset_calculator_arr, B_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *A_offset_calculator, *B_offset_calculator,
                   *out_offset_calculator;
  std::tie(A_offset_calculator_arr, A_offset_calculator) =
    AllocOffsetCalculator(inputA, stream);
  std::tie(B_offset_calculator_arr, B_offset_calculator) =
    AllocOffsetCalculator(inputB, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "WhereCuda", [&]() {
      where_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        cond->data_ptr<int64_t>(), inputA->data_ptr<spec_t>(),
        inputB->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size,
        A_offset_calculator, B_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({cond, inputA, inputB, output, A_offset_calculator_arr,
                      B_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
