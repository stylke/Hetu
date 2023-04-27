#pragma once
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template<typename spec_t>
struct kplus{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t a, spec_t b) {
    return a + b;
  }
};

template<typename spec_t>
struct kminus{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t a, spec_t b) {
    return a - b;
  }
};

template<typename spec_t>
struct kmultiplies{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t a, spec_t b) {
    return a * b;
  }
};

template<typename spec_t>
struct kdivides{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t a, spec_t b) {
    return a / b;
  }
};


} // namespace impl
} // namespace hetu
