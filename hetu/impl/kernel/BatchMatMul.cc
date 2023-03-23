#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void BatchMatMulCpu(const NDArray& a, bool trans_a, const NDArray& b,
                    bool trans_b, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  int ndim = a->ndim();
  int m = output->shape(ndim - 2);
  int n = output->shape(ndim - 1);
  int k = trans_a ? a->shape(ndim - 2) : a->shape(ndim - 1);
  int batchCount = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    HT_ASSERT(a->shape(i) == b->shape(i));
    HT_ASSERT(a->shape(i) == output->shape(i));
    batchCount *= a->shape(i);
  }

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "BatchMatMul", [&]() {
    spec_t alpha = 1, beta = 0;
    dnnl::memory::desc srcA_md, srcB_md, dst_md;
    if (!trans_a)
        srcA_md = dnnl::memory::desc({batchCount, m, k}, dnnl::memory::data_type::f32, 
                                      dnnl::memory::format_tag::abc);
    else
        srcA_md = dnnl::memory::desc({batchCount, m, k}, dnnl::memory::data_type::f32, 
                                      dnnl::memory::format_tag::acb);
    if (!trans_b)
        srcB_md = dnnl::memory::desc({batchCount, k, n}, dnnl::memory::data_type::f32, 
                                      dnnl::memory::format_tag::abc);
    else
        srcB_md = dnnl::memory::desc({batchCount, k, n}, dnnl::memory::data_type::f32, 
                                      dnnl::memory::format_tag::acb);
    dst_md = dnnl::memory::desc({batchCount, m, n}, dnnl::memory::data_type::f32, 
                                dnnl::memory::format_tag::abc);
                        
    auto srcA_mem = dnnl::memory(srcA_md, eng);
    auto srcB_mem = dnnl::memory(srcB_md, eng);
    auto dst_mem = dnnl::memory(dst_md, eng);

    hetu::omp::write_to_dnnl_memory(a->data_ptr<spec_t>(), srcA_mem);
    hetu::omp::write_to_dnnl_memory(b->data_ptr<spec_t>(), srcB_mem);

    auto Matmul_pd = dnnl::matmul::primitive_desc(eng, srcA_md, srcB_md, dst_md);
    auto Matmul = dnnl::matmul(Matmul_pd);

    Matmul.execute(engine_stream, {{DNNL_ARG_SRC, srcA_mem},
                                   {DNNL_ARG_WEIGHTS, srcB_mem},
                                   {DNNL_ARG_DST, dst_mem}});

    engine_stream.wait();
    hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
  });
}

} // namespace impl
} // namespace hetu
