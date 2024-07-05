#pragma once
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template<typename spec_t>
struct unary_cu{
  __forceinline__ __host__ __device__ 
  virtual spec_t operator() (spec_t input_);
  __forceinline__ __host__ __device__ 
  virtual spec_t compute(spec_t input_) {};
  __forceinline__ __host__ __device__
  unary_cu() {};
};

template<typename spec_t>
struct u_plus : unary_cu<spec_t>{
  spec_t value_;
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return input_ + value_;
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return input_ + value_;
  }
  __forceinline__ __host__ __device__
  u_plus() {}
  __forceinline__ __host__ __device__
  u_plus(spec_t value) {
    value_ = value;
  }
};

template<typename spec_t>
struct u_minus : unary_cu<spec_t>{
  spec_t value_;
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return value_ - input_;
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return value_ - input_;
  }
  __forceinline__ __host__ __device__
  u_minus() {}
  __forceinline__ __host__ __device__
  u_minus(spec_t value) {
    value_ = value;
  }
};

template<typename spec_t>
struct u_multiplies : unary_cu<spec_t>{
  spec_t value_;
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return input_ * value_;
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return input_ * value_;
  }
  __forceinline__ __host__ __device__
  u_multiplies() {}
  __forceinline__ __host__ __device__
  u_multiplies(spec_t value) {
    value_ = value;
  }
};

template<typename spec_t>
struct u_divides : unary_cu<spec_t>{
  spec_t value_;
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return value_ / input_;
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return value_ / input_;
  }
  __forceinline__ __host__ __device__
  u_divides() {}
  __forceinline__ __host__ __device__
  u_divides(spec_t value) {
    value_ = value;
  }
};

template<typename spec_t>
struct u_pow : unary_cu<spec_t>{
  spec_t value_;
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return hetu::cuda::cuda_pow(input_, value_);
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return hetu::cuda::cuda_pow(input_, value_);
  }
  __forceinline__ __host__ __device__
  u_pow() {}
  __forceinline__ __host__ __device__
  u_pow(spec_t value) {
    value_ = value;
  }
};

template<typename spec_t>
struct u_exp : unary_cu<spec_t>{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return hetu::cuda::cuda_exp(input_);
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return hetu::cuda::cuda_exp(input_);
  }
};

template<typename spec_t>
struct u_log : unary_cu<spec_t>{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return hetu::cuda::cuda_log(input_);
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return hetu::cuda::cuda_log(input_);
  }
};

template<typename spec_t>
struct u_sqrt : unary_cu<spec_t>{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return hetu::cuda::cuda_sqrt(input_);
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return hetu::cuda::cuda_sqrt(input_);
  }
};

template<typename spec_t>
struct u_abs : unary_cu<spec_t>{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return hetu::cuda::cuda_abs(input_);
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return hetu::cuda::cuda_abs(input_);
  }
};

template<typename spec_t>
struct u_gelu : unary_cu<spec_t>{
  __forceinline__ __host__ __device__ 
  spec_t operator() (spec_t input_) {
    return input_ * 0.5 * (1.0 + hetu::cuda::cuda_tanh(0.79788456 * input_ * (1 + 0.044715 * input_ * input_)));
  }
  __forceinline__ __host__ __device__ 
  spec_t compute(spec_t input_) {
    return input_ * 0.5 * (1.0 + hetu::cuda::cuda_tanh(0.79788456 * input_ * (1 + 0.044715 * input_ * input_)));
  }
};

} // namespace impl
} // namespace hetu
