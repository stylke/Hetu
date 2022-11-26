#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

static std::once_flag cublas_device_init_flags[HT_MAX_GPUS_COMPILE_TIME];
static cublasHandle_t cublas_handles[HT_MAX_GPUS_COMPILE_TIME];

static void InitCublas(int32_t device_id) {
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUBLAS_CALL(cublasCreate(&cublas_handles[device_id]));
  cudaStream_t cuda_stream = GetCUDAComputingStream(device_id).cuda_stream();
  if (cuda_stream)
    CUBLAS_CALL(cublasSetStream(cublas_handles[device_id], cuda_stream));
}

void InitCublasOnce(int32_t device_id) {
  std::call_once(cublas_device_init_flags[device_id], InitCublas, device_id);
}

} // namespace

cublasHandle_t GetCublasHandle(int32_t device_id) {
  InitCublasOnce(device_id);
  return cublas_handles[device_id];
}

template <>
void cublas_gemm<float>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const float* alpha, const float* A,
                        int32_t lda, const float* B, int32_t ldb,
                        const float* beta, float* C, int32_t ldc) {
  CUBLAS_CALL(cublasSgemm(cublas_handle, trans_a, trans_b, m, n, k, alpha, A,
                          lda, B, ldb, beta, C, ldc));
}

template <>
void cublas_gemm<double>(cublasHandle_t cublas_handle,
                         cublasOperation_t trans_a, cublasOperation_t trans_b,
                         int32_t m, int32_t n, int32_t k, const double* alpha,
                         const double* A, int32_t lda, const double* B,
                         int32_t ldb, const double* beta, double* C,
                         int32_t ldc) {
  CUBLAS_CALL(cublasDgemm(cublas_handle, trans_a, trans_b, m, n, k, alpha, A,
                          lda, B, ldb, beta, C, ldc));
}

template <>
void cublas_batch_gemm<float>(cublasHandle_t cublas_handle,
                              cublasOperation_t trans_a,
                              cublasOperation_t trans_b, int32_t m, int32_t n,
                              int32_t k, const float* alpha, const float* A,
                              int32_t lda, int32_t strideA, const float* B,
                              int32_t ldb, int32_t strideB, const float* beta,
                              float* C, int32_t ldc, int32_t strideC,
                              int32_t batch_count) {
  CUBLAS_CALL(cublasSgemmStridedBatched(
    cublas_handle, trans_a, trans_b, m, n, k, alpha, A, lda, strideA, B, ldb,
    strideB, beta, C, ldc, strideC, batch_count));
}

template <>
void cublas_batch_gemm<double>(cublasHandle_t cublas_handle,
                               cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int32_t m, int32_t n,
                               int32_t k, const double* alpha, const double* A,
                               int32_t lda, int32_t strideA, const double* B,
                               int32_t ldb, int32_t strideB, const double* beta,
                               double* C, int32_t ldc, int32_t strideC,
                               int32_t batch_count) {
  CUBLAS_CALL(cublasDgemmStridedBatched(
    cublas_handle, trans_a, trans_b, m, n, k, alpha, A, lda, strideA, B, ldb,
    strideB, beta, C, ldc, strideC, batch_count));
}

} // namespace impl
} // namespace hetu
