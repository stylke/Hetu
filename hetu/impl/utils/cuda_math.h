#pragma once

#include <cuda_runtime.h>

namespace hetu {
namespace cuda {

template <typename T>
__forceinline__ __device__ T cuda_log(T x) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_log<float>(float x) {
  return logf(x);
}

template <>
__forceinline__ __device__ double cuda_log<double>(double x) {
  return log(x);
}

template <typename T>
__forceinline__ __device__ T cuda_exp(T x) {
  HT_NOT_IMPLEMENTED << "cuda_exp is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_exp<float>(float x) {
  return expf(x);
}

template <>
__forceinline__ __device__ double cuda_exp<double>(double x) {
  return exp(x);
}

template <typename T>
__forceinline__ __device__ T cuda_sqrt(T x) {
  HT_NOT_IMPLEMENTED << "cuda_sqrt is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_sqrt<float>(float x) {
  return sqrtf(x);
}

template <>
__forceinline__ __device__ double cuda_sqrt<double>(double x) {
  return sqrt(x);
}

template <typename T>
__forceinline__ __device__ T cuda_rsqrt(T x) {
  HT_NOT_IMPLEMENTED << "cuda_rsqrt is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_rsqrt<float>(float x) {
  return rsqrtf(x);
}

template <>
__forceinline__ __device__ double cuda_rsqrt<double>(double x) {
  return rsqrt(x);
}

template <typename T>
__forceinline__ __device__ T cuda_sin(T x) {
  HT_NOT_IMPLEMENTED << "cuda_sin is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_sin<float>(float x) {
  return sinf(x);
}

template <>
__forceinline__ __device__ double cuda_sin<double>(double x) {
  return sin(x);
}

template <typename T>
__forceinline__ __device__ T cuda_cos(T x) {
  HT_NOT_IMPLEMENTED << "cuda_cos is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_cos<float>(float x) {
  return cosf(x);
}

template <>
__forceinline__ __device__ double cuda_cos<double>(double x) {
  return cos(x);
}

template <typename T>
__forceinline__ __device__ T cuda_tanh(T x) {
  HT_NOT_IMPLEMENTED << "cuda_tanh is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_tanh<float>(float x) {
  return tanhf(x);
}

template <>
__forceinline__ __device__ double cuda_tanh<double>(double x) {
  return tanh(x);
}

template <typename T>
__forceinline__ __device__ T cuda_pow(T x, T exponent) {
  HT_NOT_IMPLEMENTED << "cuda_pow is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ float cuda_pow<float>(float x, float exponent) {
  return powf(x, exponent);
}

template <>
__forceinline__ __device__ double cuda_pow<double>(double x, double exponent) {
  return pow(x, exponent);
}

template <typename T>
__forceinline__ __device__ void AtomicAdd(T* address, T val) {
  HT_NOT_IMPLEMENTED << "cuda_log is not implemented for type "
                     << typeid(T).name();
}

template <>
__forceinline__ __device__ void AtomicAdd<float>(float* address, float val) {
    atomicAdd(address, val);
}

template <>
__forceinline__ __device__ void AtomicAdd<double>(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    // return __longlong_as_double(old);
}

template <typename spec_t>
__forceinline__ __device__ float WarpReduceSum(spec_t val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

  if (wid == 0)
    val = WarpReduceSum(val);
}

} // namespace cuda
} // namespace hetu
