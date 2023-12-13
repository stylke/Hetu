#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"

#include <tuple>
#include <cassert>

namespace hetu {
namespace impl {

class OffsetCalculator {
 public:
  __device__ OffsetCalculator() = default;
  __device__ ~OffsetCalculator() = default;
  __host__ __device__ virtual inline size_t get(size_t linear_idx) const {
    return linear_idx;
  }
};

class StridedOffsetCalculator : public OffsetCalculator {
 public:
  __device__ StridedOffsetCalculator(int dims, const int64_t* shape, const int64_t* stride)
    : OffsetCalculator(),
    _dims(dims) {
    assert(dims <= HT_MAX_NDIM);

    for (int i = 0; i < dims; i++) {
      _shape[i] = shape[i];
      _stride[i] = stride[i];
    }
  }

  StridedOffsetCalculator(int dims, HTShape shape, HTShape stride)
    : OffsetCalculator(),
    _dims(dims) {
    HT_ASSERT(dims <= HT_MAX_NDIM)
      << "Currently we only support shape up to " << HT_MAX_NDIM
      << " dimensions. Got" << dims << ".";

    for (int i = 0; i < dims; i++) {
      _shape[i] = shape[i];
      _stride[i] = stride[i];
    }
  }
  __device__ ~StridedOffsetCalculator() = default;

  __host__ __device__ inline size_t get(size_t linear_idx) const override {
    size_t offset = 0;
    for (int i = _dims - 1; i >= 0; i--) {
      int64_t shape_i = _shape[i];
      auto div_idx = linear_idx / shape_i;
      auto mod_idx = linear_idx - div_idx * shape_i;
      offset += mod_idx * _stride[i];
      linear_idx = div_idx;
    }
    return offset;
  }
 
 protected:
  int _dims;
  int64_t _shape[HT_MAX_NDIM];
  int64_t _stride[HT_MAX_NDIM];
};

__global__ static void trivial_constructor(OffsetCalculator* dst) {
  new(dst) OffsetCalculator();
}

__global__ static void strided_constructor(StridedOffsetCalculator* dst, int dims,
                                           const int64_t* shape, const int64_t* stride) {
  new(dst) StridedOffsetCalculator(dims, shape, stride);
}

std::tuple<NDArray, OffsetCalculator*>
AllocOffsetCalculator(const NDArray& arr, const Stream& stream);

} // namespace impl
} // namespace hetu