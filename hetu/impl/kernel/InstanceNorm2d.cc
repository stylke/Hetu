#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void minus_mean_n_square_kernel1(const spec_t* in_arr,
                                 const spec_t* mean, spec_t* out_arr,
                                 int last_2dim, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    spec_t temp = in_arr[idx] - mean[idx / last_2dim];
    out_arr[idx] = temp * temp;
  }
}

template <typename spec_t>
void std_normal_transform(const spec_t* in_arr,
                          const spec_t* mean_arr,
                          const spec_t* var_arr, spec_t* out_arr,
                          int last_2dim, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last_2dim;
    out_arr[idx] =
      (in_arr[idx] - mean_arr[mo_idx]) / std::sqrt(var_arr[mo_idx] + eps);
  }
}

void InstanceNormCpu(const NDArray& in_arr, NDArray& mean_arr, NDArray& var_arr,
                     NDArray& out_arr, float eps, const Stream& stream) {

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
  size_t cpu_mem = ndim * sizeof(int);
  int* dimA = (int*) malloc(cpu_mem);
  int* strideA = (int*) malloc(cpu_mem);
  int* dimC = (int*) malloc(cpu_mem);
  int* strideC = (int*) malloc(cpu_mem);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = (int) in_arr->shape(i);
    dimC[i] = i < in_arr->ndim() - 2 ? (int) in_arr->shape(i) : 1;
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormCpu", [&]() {
      size_t size = temp_strideA * sizeof(spec_t);

      float one = 1.0f;
      float zero = 0.0f;

      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnl::memory::data_type::f32, in_arr->stride());
      auto dst_md = dnnl::memory::desc(mean_arr->shape(), dnnl::memory::data_type::f32, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(in_arr->data_ptr<spec_t>(), src_mem);
      if (in_arr->shape() == mean_arr->shape())
        hetu::omp::read_from_dnnl_memory(mean_arr->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_mean, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(mean_arr->data_ptr<spec_t>(), dst_mem);
      }

      minus_mean_n_square_kernel1<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        out_arr->data_ptr<spec_t>(), last_2dim, temp_strideA);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(out_arr->data_ptr<spec_t>(), src_mem);
      if (in_arr->shape() == mean_arr->shape())
        hetu::omp::read_from_dnnl_memory(var_arr->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_mean, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(var_arr->data_ptr<spec_t>(), dst_mem);
      }

      std_normal_transform<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        var_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), last_2dim,
        eps, temp_strideA);    
    });
  return;
}

template <typename spec_t>
void calculate_grad_kernel(const spec_t* out_grads,
                           const spec_t* in_arr,
                           const spec_t* mean_arr,
                           const spec_t* var_arr, 
                           spec_t* ds, spec_t* dbias,
                           spec_t* grad_arr,
                           size_t last2dim, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last2dim;
    spec_t tmp = (dbias[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                  (var_arr[mo_idx] + eps);
    grad_arr[idx] = out_grads[idx] /std::sqrt(var_arr[mo_idx] + eps) +
      ((tmp - dbias[mo_idx]) / (spec_t)last2dim) / 
      std::sqrt(var_arr[mo_idx] + eps);
  }
}

void InstanceNormGradientCpu(const NDArray& out_grads, const NDArray& in_arr,
                             NDArray& grad_arr, const NDArray& mean_arr,
                             const NDArray& var_arr, float eps,
                             const Stream& stream) {
  int ndim = out_grads->ndim();
  HT_ASSERT(ndim == 4);
  size_t total_elements = 1;

  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
  size_t cpu_mem = ndim * sizeof(int);
  int* dimA = (int*) malloc(cpu_mem);
  int* strideA = (int*) malloc(cpu_mem);
  int* dimC = (int*) malloc(cpu_mem);
  int* strideC = (int*) malloc(cpu_mem);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = (int) in_arr->shape(i);
    dimC[i] = i < in_arr->ndim() - 2 ? (int) in_arr->shape(i) : 1;
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

  size_t size = total_elements;
  if (size == 0)
    return;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormGradientCpu", [&]() {
      spec_t* dscale = NULL;
      DataPtr dscale_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
      dscale = (spec_t*) dscale_ptr.ptr;

      spec_t* dbias = NULL;
      DataPtr dbias_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
      dbias = (spec_t*) dbias_ptr.ptr;

      spec_t* dy_mul_x = NULL;
      DataPtr dy_mul_x_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(spec_t));
      dy_mul_x = (spec_t*) dy_mul_x_ptr.ptr;

      spec_t* workspace = NULL;
      DataPtr workspace_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(spec_t));
      workspace = (spec_t*) workspace_ptr.ptr;

      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnl::memory::data_type::f32, in_arr->stride());
      auto dst_md = dnnl::memory::desc(mean_arr->shape(), dnnl::memory::data_type::f32, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(out_grads->data_ptr<spec_t>(), src_mem);
      if (in_arr->shape() == mean_arr->shape())
        hetu::omp::read_from_dnnl_memory(dbias, src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(dbias, dst_mem);
      } 
      // Create src memory objects.
      auto src_A_mem = dnnl::memory(src_md, eng);
      auto src_B_mem = dnnl::memory(src_md, eng);
      auto mdst_mem = dnnl::memory(src_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(out_grads->data_ptr<spec_t>(), src_A_mem);
      hetu::omp::write_to_dnnl_memory(in_arr->data_ptr<spec_t>(), src_B_mem);

      auto binary_pd = dnnl::binary::primitive_desc(eng, dnnl::algorithm::binary_mul,
                                                    src_md, src_md, src_md);

      // Create the primitive.
      auto binary_prim = dnnl::binary(binary_pd);

      // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC_0, src_A_mem});
      binary_args.insert({DNNL_ARG_SRC_1, src_B_mem});
      binary_args.insert({DNNL_ARG_DST, mdst_mem});

      // Primitive execution: binary with ReLU.
      binary_prim.execute(engine_stream, binary_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from memory object's handle.
      hetu::omp::read_from_dnnl_memory(dy_mul_x, mdst_mem);
      
      hetu::omp::write_to_dnnl_memory(dy_mul_x, src_mem);
      if (in_arr->shape() == mean_arr->shape())
        hetu::omp::read_from_dnnl_memory(dscale, src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(dscale, dst_mem);
      } 
      calculate_grad_kernel<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        dscale, dbias,
        grad_arr->data_ptr<spec_t>(), last2dim, eps, size);

      FreeToMemoryPool(dscale_ptr);
      FreeToMemoryPool(dbias_ptr);
      FreeToMemoryPool(dy_mul_x_ptr);
      FreeToMemoryPool(workspace_ptr);
    }); 
  free(dimA);
  free(strideA);
  free(dimC);
  free(strideC);
}

} // namespace impl
} // namespace hetu
