#pragma once

#include "hetu/impl/utils/cuda_utils.h"
#include <cublas_v2.h>

namespace hetu {
namespace impl {

cublasHandle_t GetCublasHandle(int32_t device_id);

template <typename T>
inline void cublas_gemm(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const T* alpha, const T* A, int32_t lda,
                        const T* B, int32_t ldb, const T* beta, T* C,
                        int32_t ldc) {
  HT_NOT_IMPLEMENTED << "cublas_gemm is not implemented for type "
                     << typeid(T).name();
}

template <>
void cublas_gemm<float>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const float* alpha, const float* A,
                        int32_t lda, const float* B, int32_t ldb,
                        const float* beta, float* C, int32_t ldc);

template <>
void cublas_gemm<double>(cublasHandle_t cublas_handle,
                         cublasOperation_t trans_a, cublasOperation_t trans_b,
                         int32_t m, int32_t n, int32_t k, const double* alpha,
                         const double* A, int32_t lda, const double* B,
                         int32_t ldb, const double* beta, double* C,
                         int32_t ldc);

template <typename T>
inline void
cublas_batch_gemm(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                  cublasOperation_t trans_b, int32_t m, int32_t n, int32_t k,
                  const T* alpha, const T* A, int32_t lda, int32_t strideA,
                  const T* B, int32_t ldb, int32_t strideB, const T* beta, T* C,
                  int32_t ldc, int32_t strideC, int32_t batch_count) {}

template <>
void cublas_batch_gemm<float>(cublasHandle_t cublas_handle,
                              cublasOperation_t trans_a,
                              cublasOperation_t trans_b, int32_t m, int32_t n,
                              int32_t k, const float* alpha, const float* A,
                              int32_t lda, int32_t strideA, const float* B,
                              int32_t ldb, int32_t strideB, const float* beta,
                              float* C, int32_t ldc, int32_t strideC,
                              int32_t batch_count);

template <>
void cublas_batch_gemm<double>(cublasHandle_t cublas_handle,
                               cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int32_t m, int32_t n,
                               int32_t k, const double* alpha, const double* A,
                               int32_t lda, int32_t strideA, const double* B,
                               int32_t ldb, int32_t strideB, const double* beta,
                               double* C, int32_t ldc, int32_t strideC,
                               int32_t batch_count);

} // namespace impl
} // namespace hetu

namespace {
inline const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_UNKNOWN_ERROR";
  }
}
} // namespace

#define CUBLAS_CALL(f)                                                         \
  for (cublasStatus_t status = (f); status != CUBLAS_STATUS_SUCCESS;           \
       status = CUBLAS_STATUS_SUCCESS)                                         \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Cublas call " << #f << " failed: " << cublasGetErrorString(status)
