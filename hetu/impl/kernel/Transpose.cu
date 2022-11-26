#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

const int TILE_SIZE = 32;
const int BLOCK_ROWS = 8;

template <typename spec_t>
__global__ void batch_transpose_kernel(const spec_t* input, spec_t* output,
                                       int64_t rows, int64_t cols,
                                       int64_t num_tile_rows,
                                       int64_t num_tile_cols,
                                       int64_t block_nums, int64_t tile_size) {
  const int64_t src_rows = rows;
  const int64_t src_cols = cols;
  const int64_t dst_rows = cols;
  const int64_t dst_cols = rows;
  __shared__ spec_t tile[TILE_SIZE][TILE_SIZE + 1]; // To avoid bank conflict.

  int64_t batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) {
    const int64_t batch_index = i / batch_num_tile; // the index of batch.
    const int64_t tile_index = i -
      batch_index *
        batch_num_tile; // equal to i % (num_tile_rows*num_tile_cols). the
                        // flatten index of tile in a batch.

    const int64_t tile_row_index =
      tile_index / num_tile_cols; // the row index of tile in a batch.
    const int64_t tile_col_index = tile_index -
      tile_row_index * num_tile_cols; // equal to k % num_tile_cols. the col
                                      // index of tile in a batch.

    const int64_t offset = batch_index * src_rows * src_cols;
    {
      int64_t col_in_tile = threadIdx.x;
      int64_t col_in_matrix = tile_col_index * tile_size + threadIdx.x;
#pragma unroll
      for (int64_t row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += BLOCK_ROWS) {
        int64_t row_in_matrix = row_in_tile + tile_row_index * tile_size;
        if (col_in_matrix < src_cols && row_in_matrix < src_rows) {
          tile[row_in_tile][col_in_tile] =
            input[offset + row_in_matrix * src_cols + col_in_matrix];
        }
      }
    }
    __syncthreads();
    {
      int64_t col_in_tile = threadIdx.x;
      int64_t col_in_matrix = tile_row_index * tile_size + threadIdx.x;
#pragma unroll
      for (int64_t row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += BLOCK_ROWS) {
        int64_t row_in_matrix = row_in_tile + tile_col_index * tile_size;
        if (col_in_matrix < dst_cols && row_in_matrix < dst_rows) {
          output[offset + row_in_matrix * dst_cols + col_in_matrix] =
            tile[col_in_tile][row_in_tile];
        }
      }
    }
    __syncthreads();
  }
}

template <typename spec_t>
__global__ void transpose_kernel(const spec_t* input, spec_t* output,
                                 const uint* buf, const uint ndims,
                                 size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  const uint* in_strides = buf;
  const uint* out_strides = buf + ndims;
  const uint* perm = buf + ndims * 2;
  uint i_idx = 0;
  uint t = idx;
#pragma unroll
  for (int i = 0; i < ndims; ++i) {
    const uint ratio = t / out_strides[i];
    t -= ratio * out_strides[i];
    i_idx += ratio * in_strides[perm[i]];
  }
  output[idx] = input[i_idx];
}

bool BatchTranspose(size_t ndims, int64_t* perm) {
  for (int i = 0; i < ndims - 2; ++i) {
    if (perm[i] != i)
      return false;
  }
  if (perm[ndims - 1] == ndims - 2 && perm[ndims - 2] == ndims - 1)
    return true;
  return false;
}

void TransposeCuda(const NDArray& input, NDArray& output, int64_t* perm,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  uint ndim = uint(input->ndim());
  uint ndim_ = uint(output->ndim());
  HT_ASSERT(ndim == ndim_);
  if (BatchTranspose(ndim, perm)) {
    int64_t rows = input->shape(ndim - 2);
    int64_t cols = input->shape(ndim - 1);
    int64_t tile_size = TILE_SIZE;
    int64_t num_tile_rows = (rows + tile_size - 1) / tile_size;
    int64_t num_tile_cols = (cols + tile_size - 1) / tile_size;
    int64_t num_batches = input->numel() / (rows * cols);
    int64_t block_nums = num_batches * num_tile_rows * num_tile_cols;
    CUDAStream cuda_stream(stream);
    int dev_id = cuda_stream.device_id();
    dim3 blocks, threads;
    threads.x = TILE_SIZE;
    threads.y = BLOCK_ROWS;
    blocks.x = block_nums;
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "TransposeCuda", [&]() {
        batch_transpose_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), rows, cols,
          num_tile_rows, num_tile_cols, block_nums, tile_size);
      });
  } else {
    const int64_t* in_dims = input->shape().data();
    const int64_t* out_dims = output->shape().data();
    uint* buf = (uint*) malloc(3 * ndim * sizeof(uint));
    uint* gpu_buf = NULL;

    uint in_stride = 1;
    uint out_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      buf[i] = uint(in_stride);
      buf[ndim + i] = uint(out_stride);
      buf[ndim * 2 + i] = uint(perm[i]);
      in_stride *= uint(in_dims[i]);
      out_stride *= uint(out_dims[i]);
    }
    HT_ASSERT(in_stride == out_stride);
    size_t size = in_stride;
    CUDAStream cuda_stream(stream);
    int dev_id = cuda_stream.device_id();

    size_t buf_size = 3 * ndim * sizeof(uint);
    DataPtr gpu_buf_ptr = AllocFromMemoryPool(input->device(), buf_size);
    gpu_buf = (uint*) gpu_buf_ptr.ptr;

    if (size == 0)
      return;
    dim3 blocks, threads;
    threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    CUDA_CALL(cudaMemcpyAsync(gpu_buf, (void*) buf, buf_size,
                              cudaMemcpyHostToDevice, cuda_stream));
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "TransposeCuda", [&]() {
        transpose_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), gpu_buf, ndim,
          size);
      });
    FreeToMemoryPool(gpu_buf_ptr);
    free(buf);
  }
}

} // namespace impl
} // namespace hetu
