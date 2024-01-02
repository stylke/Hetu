#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

#define BINARYCONST(input, value, output, op, stream)                                           \
        HT_ASSERT_CUDA_DEVICE(input);                                                           \
        HT_ASSERT_SAME_DEVICE(input, output);                                                   \
        size_t size = input->numel();                                                           \
        if (size == 0)                                                                          \
          return;                                                                               \
        bool contiguous = input->is_contiguous() && output->is_contiguous();                    \
        if (contiguous) {                                                                       \
          HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                               \
            input->dtype(), spec_t, "BinaryConstCuda", [&]() {                                  \
              launch_vectorized_unary_kernel(input->data_ptr<spec_t>(), size,                   \
                                            output->data_ptr<spec_t>(), stream,                 \
                                            [=] __device__ (spec_t x) -> spec_t {               \
                                              return op<spec_t, spec_t>{}                       \
                                                     (static_cast<spec_t>(value), x);           \
                                            });                                                 \
          });                                                                                   \
        } else {                                                                                \
          constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;    \
          dim3 block(128);                                                                      \
          dim3 grid(DIVUP(size, unroll_factor * block.x));                                      \
          NDArray in_offset_calculator_arr, out_offset_calculator_arr;                          \
          OffsetCalculator *in_offset_calculator, *out_offset_calculator;                       \
          std::tie(in_offset_calculator_arr, in_offset_calculator) =                            \
            AllocOffsetCalculator(input, stream);                                               \
          std::tie(out_offset_calculator_arr, out_offset_calculator) =                          \
            AllocOffsetCalculator(output, stream);                                              \
          CUDAStream cuda_stream(stream);                                                       \
          HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                               \
            input->dtype(), spec_t, "BinaryConstCuda", [&]() {                                  \
              unary_kernel<128, unroll_factor><<<grid, block, 0, cuda_stream>>>(                \
                input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>(),                    \
                [=] __device__ (spec_t x) -> spec_t {                                           \
                  return op<spec_t, spec_t>{}                                                   \
                         (static_cast<spec_t>(value), x);                                       \
                }, in_offset_calculator, out_offset_calculator);                                \
          });                                                                                   \
          NDArray::MarkUsedBy({in_offset_calculator_arr, out_offset_calculator_arr}, stream);   \
        }                                                                                       \
        NDArray::MarkUsedBy({input, output}, stream);                 

namespace hetu {
namespace impl {

// template <typename spec_t, typename Operator>
// __global__ void binary_const_kernel(const spec_t* input, spec_t value,
//                                     size_t size, Operator op, spec_t* output,
//                                     const OffsetCalculator* in_offset_calculator,
//                                     const OffsetCalculator* out_offset_calculator) {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < size) {
//     auto in_offset = in_offset_calculator->get(idx);
//     auto out_offset = out_offset_calculator->get(idx);
//     output[out_offset] = op(value, input[in_offset]);
//   }
// }

void AddConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kplus, stream)
}

void SubConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kminus, stream)
}

void MulConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kmultiplies, stream)
}

void DivConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT(value != 0) << "Divided by 0.";
  BINARYCONST(input, value, output, kdivides, stream)
}


} // namespace impl
} // namespace hetu
