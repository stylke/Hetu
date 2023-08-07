#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/cuda/CUDABlas.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void MatVecMulCuda(const NDArray& a, bool trans, const NDArray& x,
                NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, x);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(x, 1);
  HT_ASSERT_NDIM(output, 1);
  HT_ASSERT_SAME_DTYPE(a, x);
  HT_ASSERT_SAME_DTYPE(a, output);

  cublasHandle_t cublas_handle = GetCublasHandle(output->device().index());
  hetu::cuda::CUDADeviceGuard guard(output->device().index());
  int32_t m = a->shape(1);
  int32_t n = a->shape(0);

  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatVecMul", [&]() {
    spec_t alpha = 1, beta = 0;
    cublas_gemv<spec_t>(cublas_handle, !trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                        m, n, &alpha,
                        a->data_ptr<spec_t>(), m,
                        x->data_ptr<spec_t>(), 1, &beta,
                        output->data_ptr<spec_t>(), 1);
  });
    //   HT_LOG_INFO << "_____________up____________\n"
    // << a << "\n"
    // << x << "\n" << output
    // << "\n__________down_____________";
}

} // namespace impl
} // namespace hetu
