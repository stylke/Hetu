#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "cmath"

namespace hetu {
namespace impl {

template <typename spec_t>
void pow_cpu(const spec_t* input, double exponent, size_t size,
             spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::pow(input[idx], exponent);
  }
}

void PowCpu(const NDArray& input, double exponent, NDArray& output,
            const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "PowCpu", [&]() {
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng);
      auto dst_mem = dnnl::memory(mat_md, eng);
      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);

      auto Reciprocal_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                           dnnl::algorithm::eltwise_pow, mat_md, mat_md, float(1.0), float(exponent));
      auto Reciprocal = dnnl::eltwise_forward(Reciprocal_pd);

      Reciprocal.execute(engine_stream,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      engine_stream.wait();
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

} // namespace impl
} // namespace hetu
