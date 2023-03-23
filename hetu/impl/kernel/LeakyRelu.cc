#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void leaky_relu_cpu(const spec_t* input, spec_t alpha, size_t size,
                    spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx];
    if (input[idx] < 0)
      output[idx] *= alpha;
  }
}

template <typename spec_t>
void leaky_relu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                             spec_t alpha_LeakyRelu, size_t size,
                             spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = output_grad[idx];
    if (input[idx] < 0)
      output[idx] *= alpha_LeakyRelu;
  }
}

void LeakyReluCpu(const NDArray& input, double alpha, NDArray& output,
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
    input->dtype(), spec_t, "LeakyReluCpu", [&]() {
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng);
      auto dst_mem = dnnl::memory(mat_md, eng);
      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);

      auto LeakyRelu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                           dnnl::algorithm::eltwise_relu, mat_md, mat_md, float(alpha), float(0.0));
      auto LeakyRelu = dnnl::eltwise_forward(LeakyRelu_pd);

      LeakyRelu.execute(engine_stream,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      engine_stream.wait();
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

void LeakyReluGradientCpu(const NDArray& input, const NDArray& output_grad,
                          double alpha, NDArray& input_grad,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluGradientCpu", [&]() {
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng);
      auto g_dst_mem = dnnl::memory(mat_md, eng);
      auto g_src_mem = dnnl::memory(mat_md, eng);
      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);
      hetu::omp::write_to_dnnl_memory(output_grad->data_ptr<spec_t>(), g_dst_mem);

      auto LeakyRelu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                           dnnl::algorithm::eltwise_relu, mat_md, mat_md, float(alpha), float(0.0));

      auto LeakyRelu_bwd_pd = dnnl::eltwise_backward::primitive_desc(eng,
            dnnl::algorithm::eltwise_relu, mat_md, mat_md,
            mat_md, float(alpha), float(0.0), LeakyRelu_pd);
      
      auto LeakyRelu_bwd = dnnl::eltwise_backward(LeakyRelu_bwd_pd);

      LeakyRelu_bwd.execute(engine_stream,
                      {{DNNL_ARG_SRC, src_mem}, 
                       {DNNL_ARG_DIFF_DST, g_dst_mem},
                       {DNNL_ARG_DIFF_SRC, g_src_mem}});
      engine_stream.wait();
      hetu::omp::read_from_dnnl_memory(input_grad->data_ptr<spec_t>(), g_src_mem);
    });
}

} // namespace impl
} // namespace hetu
