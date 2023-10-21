#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

<<<<<<< HEAD
template <typename spec_t>
__global__ void reduce_sum_naive_kernel(const spec_t* input, spec_t* output,
                                        size_t befor_dim_size,
                                        size_t reduce_dim_size,
                                        size_t after_dim_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= befor_dim_size * after_dim_size)
    return;  
  size_t x = idx / after_dim_size;
  size_t y = idx % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
    stride = after_dim_size;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = reduce_dim_size;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
=======
template <typename arg_t>
struct SumFunctor {
  __device__ __forceinline__ arg_t operator()(arg_t a, arg_t b) const {
    return a + b;
>>>>>>> 2c2b41a04751c35a197d821a66142997e0d95613
  }
};

void ReduceSumCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "ReduceSumCuda", [&]() {
        using arg_t = opmath_type<spec_t>;
        launch_reduce_kernel<spec_t, arg_t>(in_arr, out_arr, axes, num_ax,
                             functor_wrapper<spec_t, arg_t>(SumFunctor<arg_t>()), 0., stream);
    });
}

} // namespace impl
} // namespace hetu