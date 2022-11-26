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

} // namespace cuda
} // namespace hetu
