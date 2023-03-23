#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void tanh_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::tanh(input[idx]);
  }
}

template <typename spec_t>
void tanh_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = (1 - input[idx] * input[idx]) * output_grad[idx];
  }
}

void TanhCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TanhCpu", [&]() {
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng);
      auto dst_mem = dnnl::memory(mat_md, eng);
      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);

      auto Tanh_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                           dnnl::algorithm::eltwise_tanh, mat_md, mat_md, float(0.0), float(0.0));
      auto Tanh = dnnl::eltwise_forward(Tanh_pd);

      Tanh.execute(engine_stream,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      engine_stream.wait();
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

void TanhGradientCpu(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TanhGradientCpu", [&]() {
      tanh_gradient_cpu<spec_t>(input->data_ptr<spec_t>(),
                                output_grad->data_ptr<spec_t>(), size,
                                input_grad->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
