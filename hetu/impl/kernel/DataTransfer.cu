#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_a_t, typename spec_b_t>
__global__ void data_transfer_kernel(const spec_a_t* src, spec_b_t* dst, size_t numel,
                                     const OffsetCalculator* from_offset_calculator,
                                     const OffsetCalculator* to_offset_calculator) {
  // TODO: cuda memory access aligns to 4 bytes
  // so we should re-consider 8- or 16-bit values
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;
  auto from_offset = from_offset_calculator->get(idx);
  auto to_offset = to_offset_calculator->get(idx);
  dst[to_offset] = static_cast<spec_b_t>(src[from_offset]);
}

bool require_temp_storage(const NDArray& from, const NDArray& to) {
  auto from_device = from->device();
  auto to_device = to->device();
  if (from_device == to_device) {
    return false;
  }
  bool same_dtype = from->dtype() == to->dtype();
  if (same_dtype && from->is_contiguous() && to->is_contiguous()
   || from_device.is_cuda() && to_device.is_cuda()) {
    return false;
  } else {
    return true;
  }
}

void transfer_device_to_device(const NDArray& from, NDArray& to, const Stream& stream) {
  size_t numel = from->numel();
  CUDAStream cuda_stream(stream);

  bool memcpy_eligible = from->dtype() == to->dtype()
                      && from->is_contiguous() && to->is_contiguous();
  auto from_device = from->device();
  auto to_device = to->device();
  void* to_ptr = to->raw_data_ptr();
  void* from_ptr = from->raw_data_ptr();

  if (memcpy_eligible) {
    size_t num_bytes = numel * DataType2Size(from->dtype());
    bool require_peer_memcpy = from->device().index() != to->device().index();
    
    if (to_ptr != from_ptr || from->device() != to->device()) {
      if (require_peer_memcpy) {
       // TODO: check that the stream belongs to source GPU as recommended by
       // https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf.
        CudaMemcpyPeerAsync(to_ptr, to->device().index(),
                            from_ptr, from->device().index(),
                            num_bytes, cuda_stream);
      } else {
        CudaMemcpyAsync(to_ptr, from_ptr, num_bytes, cudaMemcpyDeviceToDevice,
                        cuda_stream);
      }
    }
  } else {
    dim3 blocks, threads;
    threads.x = MIN(numel, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(numel, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    NDArray from_offset_calculator_arr, to_offset_calculator_arr;
    OffsetCalculator *from_offset_calculator, *to_offset_calculator;
    std::tie(from_offset_calculator_arr, from_offset_calculator) =
      AllocOffsetCalculator(from, stream);
    std::tie(to_offset_calculator_arr, to_offset_calculator) = 
      AllocOffsetCalculator(to, stream);
    HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
      from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCuda",
      [&]() {
        data_transfer_kernel<spec_a_t, spec_b_t>
          <<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<spec_a_t*>(from_ptr),
            reinterpret_cast<spec_b_t*>(to_ptr), numel,
            from_offset_calculator, to_offset_calculator);
      });
    NDArray::MarkUsedBy({from_offset_calculator_arr, to_offset_calculator_arr}, stream);
  }
}

void DataTransferCuda(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_SAME_SHAPE(from, to);
  size_t numel = from->numel();
  if (numel == 0)
    return;

  if (require_temp_storage(from, to)) {
    auto from_dsize = DataType2Size(from->dtype());
    auto to_dsize = DataType2Size(to->dtype());
    NDArray from_contig, to_contig;
    if (from_dsize <= to_dsize) {
      auto from_converted = NDArray::to(from, from->device(), to->dtype(), stream.stream_index());
      from_contig = NDArray::contiguous(from_converted, stream.stream_index());
      to_contig = to->is_contiguous() ? to : NDArray::empty_like(to);
    } else {
      from_contig = NDArray::contiguous(from, stream.stream_index());
      to_contig = NDArray::empty(to->shape(), to->device(), from->dtype()); 
    }
    DataTransferCuda(from_contig, to_contig, stream);
    if (!to->is_contiguous() || to->dtype() != to_contig->dtype()) {
      HT_ASSERT(to_contig->device() == to->device());
      DataTransferCuda(to_contig, to, stream);
    }
    NDArray::MarkUsedBy({from, to}, stream);
    return;
  }

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  // Copy between GPUs
  if (from->device().is_cuda() && to->device().is_cuda()) {
    transfer_device_to_device(from, to, stream);
    NDArray::MarkUsedBy({from, to}, stream);
    return;
  }

  // Copy between CPU and GPU
  size_t num_bytes = numel * DataType2Size(from->dtype());
  cudaMemcpyKind kind;
  if (from->device().is_cuda() && to->device().is_cpu()) {
    kind = cudaMemcpyDeviceToHost;
  } else if (from->device().is_cpu() && to->device().is_cuda()) {
    kind = cudaMemcpyHostToDevice;
  } else {
    HT_RUNTIME_ERROR << "Cannot use DataTransferCuda to "
                     << "copy data between CPU tensors. "
                     << "Please use DataTransferCpu instead.";
  }
  CudaMemcpyAsync(to->raw_data_ptr(), from->raw_data_ptr(), num_bytes,
                  kind, cuda_stream);
  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
