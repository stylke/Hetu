#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_a_t, typename spec_b_t>
__global__ void data_transfer_kernel(const spec_a_t* src, spec_b_t* dst,
                                     size_t numel) {
  // TODO: cuda memory access aligns to 4 bytes
  // so we should re-consider 8- or 16-bit values
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;
  dst[idx] = src[idx];
}

void DataTransferCuda(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_COPIABLE(from, to);
  size_t numel = from->numel();
  if (numel == 0)
    return;

  if (from->dtype() != to->dtype() && from->device() != to->device()) {
    // When types are different, we could only support
    // transferring the data on the same device.
    // Hence, we make a copy if they are on different devices.
    auto from_dsize = DataType2Size(from->dtype());
    auto to_dsize = DataType2Size(to->dtype());
    if (from_dsize <= to_dsize) {
      // Do NOT call `DataTransferCuda` since `from` may be on host memory
      auto aux =
        NDArray::to(from, from->device(), to->dtype(), stream.stream_index());
      DataTransferCuda(aux, to, stream);
    } else {
      auto aux = NDArray::empty(to->shape(), to->device(), from->dtype());
      DataTransferCuda(from, aux, stream);
      // Do NOT call `DataTransferCuda` since `to` may be on host memory
      NDArray::to(aux, to->device(), to->dtype(), stream.stream_index(), to);
    }
    return;
  }

  size_t num_bytes = numel * DataType2Size(from->dtype());
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  if (from->device().is_cuda() && to->device().is_cuda()) {
    if (from->device().index() == to->device().index()) {
      void* to_ptr = to->raw_data_ptr();
      void* from_ptr = from->raw_data_ptr();
      if (to_ptr == from_ptr) {
        HT_ASSERT(from->dtype() == to->dtype())
          << "NDArrays with " << from->dtype() << " and " << to->dtype()
          << " types are sharing the same storage, which is not allowed.";
        return;
      }
      if (from->dtype() == to->dtype()) {
        CudaMemcpyAsync(to_ptr, from_ptr, num_bytes, cudaMemcpyDeviceToDevice,
                        cuda_stream);
      } else {
        dim3 blocks, threads;
        threads.x = MIN(numel, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
        blocks.x = DIVUP(numel, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
        HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
          from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu",
          [&]() {
            data_transfer_kernel<spec_a_t, spec_b_t>
              <<<blocks, threads, 0, cuda_stream>>>(
                reinterpret_cast<spec_a_t*>(from_ptr),
                reinterpret_cast<spec_b_t*>(to_ptr), numel);
          });
      }
    } else {
      // TODO: check that the stream belongs to source GPU as recommended by
      // https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf.
      CudaMemcpyPeerAsync(to->raw_data_ptr(), to->device().index(),
                          from->raw_data_ptr(), from->device().index(),
                          num_bytes, cuda_stream);
    }
  } else if (from->device().is_cuda() && to->device().is_cpu()) {
    CudaMemcpyAsync(to->raw_data_ptr(), from->raw_data_ptr(), num_bytes,
                    cudaMemcpyDeviceToHost, cuda_stream);
  } else if (from->device().is_cpu() && to->device().is_cuda()) {
    CudaMemcpyAsync(to->raw_data_ptr(), from->raw_data_ptr(), num_bytes,
                    cudaMemcpyHostToDevice, cuda_stream);
  } else {
    HT_RUNTIME_ERROR << "Cannot use DataTransferCuda to "
                     << "copy data between CPU tensors. "
                     << "Please use DataTransferCpu instead.";
  }
}

} // namespace impl
} // namespace hetu
