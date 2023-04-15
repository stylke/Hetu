#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

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
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, alpha]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());\
      dnnl::stream engine_stream(eng);
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

      auto LeakyRelu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                           dnnl::algorithm::eltwise_relu, mat_md, mat_md, float(alpha), float(0.0));
      auto LeakyRelu = dnnl::eltwise_forward(LeakyRelu_pd);

      LeakyRelu.execute(engine_stream,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      engine_stream.wait();
      },"LeakyRelu");
      //cpu_stream.Sync();
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
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [stream, output_grad, input, input_grad, alpha]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());\
      dnnl::stream engine_stream(eng);
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto g_dst_mem = dnnl::memory(mat_md, eng, output_grad->data_ptr<spec_t>());
      auto g_src_mem = dnnl::memory(mat_md, eng, input_grad->data_ptr<spec_t>());

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
      },"LeakyReluGradient");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
