#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void MatMulCuda(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
                NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());
  int32_t m = output->shape(1);
  int32_t n = output->shape(0);
  int32_t k = trans_a ? a->shape(0) : a->shape(1);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatMul", [&]() {
    spec_t alpha = 1, beta = 0;
    float alpha_f = 1, beta_f = 0;
    if (output->dtype() == DataType::FLOAT16 || output->dtype() == DataType::BFLOAT16) {
      cublas_gemm<spec_t>(cublas_handle, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha_f),
                          b->data_ptr<spec_t>(), trans_b ? k : m,
                          a->data_ptr<spec_t>(), trans_a ? n : k, static_cast<const void*>(&beta_f),
                          output->data_ptr<spec_t>(), m);
    }
    else {
      cublas_gemm<spec_t>(cublas_handle, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, static_cast<const void*>(&alpha),
                          b->data_ptr<spec_t>(), trans_b ? k : m,
                          a->data_ptr<spec_t>(), trans_a ? n : k, static_cast<const void*>(&beta),
                          output->data_ptr<spec_t>(), m);
    }
  });
  NDArray::MarkUsedBy({a, b, output}, stream);
}

} // namespace impl
} // namespace hetu
