#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

template <typename spec_a_t, typename spec_b_t, typename Operator>
__global__ void binary_elewise_broadcast_kernel(
  const spec_a_t* inputA, const spec_b_t* inputB, size_t size, Operator op,
  spec_a_t* output, const int64_t* A_dims, const int64_t* B_dims,
  size_t A_ndims, size_t B_ndims, const int64_t* out_strides, size_t out_dims,
  const OffsetCalculator* A_offset_calculator,
  const OffsetCalculator* B_offset_calculator,
  const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t A_ind = 0;
  size_t temp = idx;
  for (size_t i = 0; i < out_dims; ++i) {
    A_ind *= A_dims[i];
    A_ind += (A_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  size_t B_ind = 0;
  temp = idx;
  for (size_t i = 0; i < out_dims; ++i) {
    B_ind *= B_dims[i];
    B_ind += (B_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  auto A_offset = A_offset_calculator->get(A_ind);
  auto B_offset = B_offset_calculator->get(B_ind);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = op(inputA[A_offset], inputB[B_offset]);
}

#define BinaryElewiseCudaHelper(inputA, inputB, output, op, stream, name)      \
  do {                                                                         \
    HT_ASSERT_CUDA_DEVICE(inputA);                                             \
    HT_ASSERT_SAME_DEVICE(inputA, output);                                     \
    HT_ASSERT_SAME_DEVICE(inputB, output);                                     \
    size_t sizeA = inputA->numel();                                            \
    size_t sizeB = inputB->numel();                                            \
    if (sizeA == sizeB) {                                                      \
      auto size = sizeA;                                                       \
      if (inputA->dtype() == inputB->dtype()) {                                \
        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                \
          inputA->dtype(), spec_t, name, [&]() {                               \
            launch_loop_kernel<spec_t, spec_t, spec_t>(                        \
                inputA, inputB, output, size, stream,                          \
                op<spec_t, spec_t>());                                         \
          });                                                                  \
        NDArray::MarkUsedBy(                                                   \
          {inputA, inputB, output}, stream);                                   \
      } else {                                                                 \
        HT_NOT_IMPLEMENTED                                                     \
          << name << " across different data types is not supported yet";      \
      }                                                                        \
    } else {                                                                   \
      size_t size = output->numel();                                           \
      size_t output_dim = output->ndim();                                      \
      HTShape A_dims(output_dim);                                              \
      HTShape B_dims(output_dim);                                              \
      HTStride out_strides(output_dim);                                        \
      size_t output_size = 1;                                                  \
      size_t A_diff = output_dim - inputA->ndim();                             \
      size_t B_diff = output_dim - inputB->ndim();                             \
      for (int i = static_cast<int>(output_dim) - 1; i >= 0; --i) {            \
        out_strides[i] = output_size;                                          \
        output_size *= output->shape(i);                                       \
        A_dims[i] = i < A_diff ? 1 : inputA->shape(i - A_diff);                \
        B_dims[i] = i < B_diff ? 1 : inputB->shape(i - B_diff);                \
      }                                                                        \
      auto device_id = output->device().index();                               \
      hetu::cuda::CUDADeviceGuard guard(device_id);                            \
      NDArray A_offset_calculator_arr, B_offset_calculator_arr,                \
              out_offset_calculator_arr;                                       \
      OffsetCalculator *A_offset_calculator, *B_offset_calculator,             \
                       *out_offset_calculator;                                 \
      std::tie(A_offset_calculator_arr, A_offset_calculator) =                 \
        AllocOffsetCalculator(inputA, stream);                                 \
      std::tie(B_offset_calculator_arr, B_offset_calculator) =                 \
        AllocOffsetCalculator(inputB, stream);                                 \
      std::tie(out_offset_calculator_arr, out_offset_calculator) =             \
        AllocOffsetCalculator(output, stream);                                 \
      CUDAStream cuda_stream(stream);                                          \
      auto A_dims_arr = hetu::cuda::to_int64_ndarray(A_dims, device_id);       \
      auto B_dims_arr = hetu::cuda::to_int64_ndarray(B_dims, device_id);       \
      auto out_strides_arr =                                                   \
        hetu::cuda::to_int64_ndarray(out_strides, device_id);                  \
      dim3 blocks, threads;                                                    \
      threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);                 \
      blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);                \
      if (inputA->dtype() == inputB->dtype()) {                                \
        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                \
          inputA->dtype(), spec_t, name, [&]() {                               \
            binary_elewise_broadcast_kernel<spec_t, spec_t>                    \
              <<<blocks, threads, 0, cuda_stream>>>(                           \
                inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size,  \
                op<spec_t, spec_t>(), output->data_ptr<spec_t>(),              \
                A_dims_arr->data_ptr<int64_t>(),                               \
                B_dims_arr->data_ptr<int64_t>(), inputA->ndim(),               \
                inputB->ndim(), out_strides_arr->data_ptr<int64_t>(),          \
                output_dim, A_offset_calculator, B_offset_calculator,          \
                out_offset_calculator);                                        \
          });                                                                  \
        NDArray::MarkUsedBy(                                                   \
          {inputA, inputB, output, A_dims_arr, B_dims_arr, out_strides_arr,    \
          A_offset_calculator_arr, B_offset_calculator_arr,                    \
          out_offset_calculator_arr}, stream);                                 \
      } else {                                                                 \
        HT_NOT_IMPLEMENTED                                                     \
          << name << " across different data types is not supported yet";      \
      }                                                                        \
    }                                                                          \
  } while (0)

void AddElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kplus, stream,
                          "AddElewiseCuda");
}

void SubElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kminus, stream,
                          "SubElewiseCuda");
}

void MulElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kmultiplies, stream,
                          "MulElewiseCuda");
}

void DivElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kdivides, stream,
                          "DivElewiseCuda");
}

} // namespace impl
} // namespace hetu
