#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

static constexpr uint32_t NUM_THREADS = HT_WARP_SIZE * 4;
static constexpr uint32_t THREAD_WORK_SIZE = 4;
static constexpr uint32_t BLOCK_WORK_SIZE = NUM_THREADS * THREAD_WORK_SIZE;

template <typename spec_t>
inline int get_vectorize_size(spec_t* ptr) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<spec_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<spec_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

/////////////////////////////
/* Vectorized Unary kernel */
/////////////////////////////
template <int vec_size, typename spec_a_t, typename spec_b_t, typename func_t>
__global__ void vectorize_unary_kernel(const spec_a_t* input, size_t size,
                                       spec_b_t* output, func_t op) {
  int remaining = size - BLOCK_WORK_SIZE * blockIdx.x;
  int idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  spec_b_t results[THREAD_WORK_SIZE];
  spec_a_t args[THREAD_WORK_SIZE];
  if (remaining < BLOCK_WORK_SIZE) {
    // do a naive unrolled loop to handle the reminder
    // load
    int base_idx = BLOCK_WORK_SIZE * idx;
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (thread_idx >= remaining) {
        break;
      }
      int linear_idx = thread_idx + base_idx;
      args[i] = input[linear_idx];
      thread_idx += NUM_THREADS;
    }
    // compute
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if ((threadIdx.x + i * NUM_THREADS) < remaining) {
        results[i] = op(args[i]);
      }
    }
    // store
    thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (thread_idx >= remaining) {
        break;
      }
      int linear_idx = thread_idx + base_idx;
      output[linear_idx] = results[i];
      thread_idx += NUM_THREADS;
    }
  } else {
    // use vectorize memory access to handle a full `block_work_size` data
    // load
    using vec_a_t = aligned_vector<spec_a_t, vec_size>;
    int loop_size = THREAD_WORK_SIZE / vec_size;
    int base_idx = BLOCK_WORK_SIZE * idx;
    auto ptr = const_cast<spec_a_t*>(input) + base_idx;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * NUM_THREADS;
      auto *from = reinterpret_cast<const vec_a_t*>(ptr);
      auto v = from[index];
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        args[i * vec_size + j] = v.val[j];
      }
    }
    // compute
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      results[i] = op(args[i]);
    }
    // store
    using vec_b_t = aligned_vector<spec_b_t, vec_size>;
    vec_b_t* to = reinterpret_cast<vec_b_t*>(output + base_idx);
    thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * NUM_THREADS;
      vec_b_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = results[i * vec_size + j];
      }
      to[index] = v;
    }
  }
}

template <typename spec_a_t, typename spec_b_t, typename func_t>
__global__ void unroll_unary_kernel(const spec_a_t* input, size_t size,
                                    spec_b_t* output, func_t op) {
  int remaining = size - BLOCK_WORK_SIZE * blockIdx.x;
  int idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  spec_b_t results[THREAD_WORK_SIZE];
  spec_a_t args[THREAD_WORK_SIZE];
  // load
  int base_idx = BLOCK_WORK_SIZE * idx;
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int linear_idx = thread_idx + base_idx;
    args[i] = input[linear_idx];
    thread_idx += NUM_THREADS;
  }
  // compute
  #pragma unroll
  for (int i = 0; i < NUM_THREADS; i++) {
    if ((threadIdx.x + i * NUM_THREADS) < remaining) {
      results[i] = op(args[i]);
    }
  }
  // store
  thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int linear_idx = thread_idx + base_idx;
    output[linear_idx] = results[i];
    thread_idx += NUM_THREADS;
  }
}

template <int nt, int vt, typename spec_a_t, typename spec_b_t, typename func_t>
__global__ void unary_kernel(const spec_a_t* input, size_t size,
                             spec_b_t* output, func_t op,
                             const OffsetCalculator* in_offset_calculator,
                             const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto in_offset = in_offset_calculator->get(idx);
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(input[in_offset]);
      idx += nt;
    }
  }
}

template <typename spec_a_t, typename spec_b_t, typename func_t>
static inline void launch_vectorized_unary_kernel(spec_a_t* input, size_t size,
                                                  spec_b_t* output, const Stream& stream,
                                                  const func_t& op) {
  int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int vec_size = std::min(get_vectorize_size<spec_a_t>(input),
                          get_vectorize_size<spec_b_t>(output));
  switch (vec_size) {
    case 4:
      vectorize_unary_kernel<4, spec_a_t, spec_b_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        input, size, output, op);
      break;
    case 2:
      vectorize_unary_kernel<2, spec_a_t, spec_b_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        input, size, output, op);
      break;
    case 1:
      unroll_unary_kernel<spec_a_t, spec_b_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        input, size, output, op);
      break;
    default:
      HT_RUNTIME_ERROR << "Unexpected vectorization size";
      __builtin_unreachable();
  }
}

///////////////////////////
/* Vectorized Set kernel */
///////////////////////////
template <int vec_size, typename spec_t, typename func_t>
__global__ void vectorize_set_kernel(size_t size, spec_t* output, func_t op) {
  int remaining = size - BLOCK_WORK_SIZE * blockIdx.x;
  int idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  spec_t results[THREAD_WORK_SIZE];
  int args[THREAD_WORK_SIZE];
  if (remaining < BLOCK_WORK_SIZE) {
    // do a naive unrolled loop to handle the reminder
    // load
    int base_idx = BLOCK_WORK_SIZE * idx;
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (thread_idx >= remaining) {
        break;
      }
      int linear_idx = thread_idx + base_idx;
      args[i] = linear_idx;
      thread_idx += NUM_THREADS;
    }
    // compute
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if ((threadIdx.x + i * NUM_THREADS) < remaining) {
        results[i] = op(args[i]);
      }
    }
    // store
    thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (thread_idx >= remaining) {
        break;
      }
      int linear_idx = thread_idx + base_idx;
      output[linear_idx] = results[i];
      thread_idx += NUM_THREADS;
    }
  } else {
    // use vectorize memory access to handle a full `block_work_size` data
    // load
    int loop_size = THREAD_WORK_SIZE / vec_size;
    int base_idx = BLOCK_WORK_SIZE * idx;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = base_idx + thread_idx + i * NUM_THREADS;
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        args[i * vec_size + j] = index + j;
      }
    }
    // compute
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      results[i] = op(args[i]);
    }
    // store
    using vec_t = aligned_vector<spec_t, vec_size>;
    vec_t* to = reinterpret_cast<vec_t*>(output + base_idx);
    thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * NUM_THREADS;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = results[i * vec_size + j];
      }
      to[index] = v;
    }
  }
}

template <typename spec_t, typename func_t>
__global__ void unroll_set_kernel(size_t size, spec_t* output, func_t op) {
  int remaining = size - BLOCK_WORK_SIZE * blockIdx.x;
  int idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  spec_t results[THREAD_WORK_SIZE];
  int args[THREAD_WORK_SIZE];
  // load
  int base_idx = BLOCK_WORK_SIZE * idx;
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int linear_idx = thread_idx + base_idx;
    args[i] = linear_idx;
    thread_idx += NUM_THREADS;
  }
  // compute
  #pragma unroll
  for (int i = 0; i < NUM_THREADS; i++) {
    if ((threadIdx.x + i * NUM_THREADS) < remaining) {
      results[i] = op(args[i]);
    }
  }
  // store
  thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int linear_idx = thread_idx + base_idx;
    output[linear_idx] = results[i];
    thread_idx += NUM_THREADS;
  }
}

template <int nt, int vt, typename spec_t, typename func_t>
__global__ void set_kernel(size_t size, spec_t* output, func_t op,
                           const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(idx);
      idx += nt;
    }
  }
}

template <typename spec_t, typename func_t>
static inline void launch_vectorized_set_kernel(size_t size, spec_t* output,
                                                const Stream& stream, const func_t& op) {
  int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int vec_size = get_vectorize_size<spec_t>(output);
  switch (vec_size) {
    case 4:
      vectorize_set_kernel<4, spec_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        size, output, op);
      break;
    case 2:
      vectorize_set_kernel<2, spec_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        size, output, op);
      break;
    case 1:
      unroll_set_kernel<spec_t, func_t><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        size, output, op);
      break;
    default:
      HT_RUNTIME_ERROR << "Unexpected vectorization size";
      __builtin_unreachable();
  }
}

} // namespace impl
} // namespace hetu
