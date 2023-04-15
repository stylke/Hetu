#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void sigmoid_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = 1.0 / (1.0 + 1.0 / std::exp(input[idx]));
  }
}

void SigmoidCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (size == 0)
    return;

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SigmoidCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
          auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

          auto Sigmoid_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_logistic, mat_md, mat_md, float(0.0), float(0.0));
          auto Sigmoid = dnnl::eltwise_forward(Sigmoid_pd);

          std::unordered_map<int, dnnl::memory> sigmoid_args;
          sigmoid_args.insert({DNNL_ARG_SRC, src_mem});
          sigmoid_args.insert({DNNL_ARG_DST, dst_mem});
          dnnl::stream engine_stream(eng);
          Sigmoid.execute(engine_stream, sigmoid_args);
          engine_stream.wait();
      }, "Sigmoid");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
