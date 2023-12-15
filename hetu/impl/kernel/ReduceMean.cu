#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

template <typename spec_t, typename arg_t>
struct MeanOps {
  arg_t factor;
  MeanOps(arg_t factor) : factor(factor) {}

  inline __device__ spec_t project(arg_t val) const {
    return static_cast<spec_t>(val * factor);
  }

  __device__ arg_t reduce(arg_t acc, arg_t val) const {
    return acc + val;
  }
};

void ReduceMeanCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  size_t reduce_num = 1;
  for (int64_t i = 0; i < num_ax; i++) {
    reduce_num *= in_arr->shape(axes[i]);
  }
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "ReduceMeanCuda", [&]() {
        using arg_t = opmath_type<spec_t>;
        using MeanFunctor = MeanOps<spec_t, arg_t>;
        arg_t factor = static_cast<arg_t>(1.0 / reduce_num);
        // TODO: Optimize it. Small size memcpy is inefficient.
        MeanFunctor ops{factor};
        auto device_id = in_arr->device().index();
        hetu::cuda::CUDADeviceGuard guard(device_id);
        CUDAStream cuda_stream(stream);
        auto ops_arr = 
          hetu::cuda::to_byte_ndarray(reinterpret_cast<uint8_t*>(&ops),
                                      sizeof(MeanFunctor), device_id);
        launch_reduce_kernel<spec_t, arg_t>(in_arr, out_arr, axes, num_ax,
                                            *(ops_arr->data_ptr<MeanFunctor>()), 0., stream);
        NDArray::MarkUsedBy({ops_arr}, stream);
    });
}

} // namespace impl
} // namespace hetu
