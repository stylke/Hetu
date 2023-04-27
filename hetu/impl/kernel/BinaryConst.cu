#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"

namespace hetu {
namespace impl {

template <typename spec_t, typename Operator>
__global__ void binary_const_kernel(const spec_t* input, spec_t value,
                                    size_t size, Operator op, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = op(value, input[idx]);
}

template<typename Operator>
void BinaryConstToolCuda(const NDArray& input, double value,
                           NDArray& output, Operator op, const Stream& stream) {
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
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BinaryConstCuda", [&]() {
      binary_const_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, op,
        output->data_ptr<spec_t>());
    });
}

void AddConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "AddConstCuda", [&]() {
        auto op = kplus<spec_t>();
        BinaryConstToolCuda(input, value, output, op, stream);
      }); 
}

void SubConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "SubConstCuda", [&]() {
        auto op = kminus<spec_t>();
        BinaryConstToolCuda(input, value, output, op, stream);
      }); 
}

void MulConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "MulConstCuda", [&]() {
        auto op = kmultiplies<spec_t>();
        BinaryConstToolCuda(input, value, output, op, stream);
      }); 
}

void DivConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "DivConstCuda", [&]() {
        auto op = kdivides<spec_t>();
        BinaryConstToolCuda(input, value, output, op, stream);
      }); 
}


} // namespace impl
} // namespace hetu
