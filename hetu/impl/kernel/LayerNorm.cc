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
                                 int last_dims, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    spec_t temp = in_arr[idx] - mean[idx / last_dims];
    out_arr[idx] = temp * temp;
  }
}

template <typename spec_t>
void layer_norm_kernel(const spec_t* in_arr, const spec_t* mean_arr, const spec_t* var_arr, 
                       const spec_t* scale,const spec_t* bias, spec_t* out_arr,
                       int last_dims, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last_dims;
    size_t add_idx = idx % last_dims;
    out_arr[idx] =
      (in_arr[idx] - mean_arr[mo_idx]) / std::sqrt(var_arr[mo_idx] + eps) * scale[add_idx] + bias[add_idx];
  }
}

void LayerNormCpu(const NDArray& in_arr, const NDArray& ln_scale,
                  const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                  NDArray& out_arr, int64_t reduce_dims, 
                  float eps, const Stream& stream) {
  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_dims = 1;
  size_t cpu_mem = ndim * sizeof(int);
  int* dimA = (int*) malloc(cpu_mem);
  int* strideA = (int*) malloc(cpu_mem);
  int* dimC = (int*) malloc(cpu_mem);
  int* strideC = (int*) malloc(cpu_mem);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = (int) in_arr->shape(i);
    dimC[i] = i < in_arr->ndim() - reduce_dims ? (int) in_arr->shape(i) : 1;
    if (i >= ndim - reduce_dims)
      last_dims *= in_arr->shape(i);
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormCpu", [&]() {
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
        out_arr->data_ptr<spec_t>(), last_dims, temp_strideA);

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

      layer_norm_kernel<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),  
        ln_scale->data_ptr<spec_t>(), ln_bias->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), 
        last_dims, eps, temp_strideA);          
    });
  return;
}

template <typename spec_t>
void calculate_gscale(const spec_t* grads, const spec_t* in_arr,
                      const spec_t* mean_arr, const spec_t* var_arr,
                      spec_t* grad_scale, spec_t eps,
                      int last_dim, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    int mo_ind = idx / last_dim;
    spec_t std = sqrtf(var_arr[mo_ind] + eps);
    spec_t x_centered = in_arr[idx] - mean_arr[mo_ind];
    spec_t x_norm = x_centered / std;
    grad_scale[idx] = grads[idx] * x_norm;
  }
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


void LayerNormGradientCpu(const NDArray& out_grads, const NDArray& in_arr,
                          const NDArray& ln_scale, NDArray& grad_arr,
                          NDArray& grad_scale, NDArray& grad_bias,
                          const NDArray& mean_arr, const NDArray& var_arr,
                          int64_t reduce_dims, float eps, const Stream& stream) {
  int ndim = out_grads->ndim();
  size_t total_elements = 1;

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdims = 1;
  for (size_t i = 0; i < reduce_dims; ++i) {
    lastdims *= out_grads->shape(ndim - 1 -i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormGradientCpu", [&]() {
      spec_t* ds = NULL;
      DataPtr ds_ptr = AllocFromMemoryPool(in_arr->device(), mean_arr->numel() * sizeof(spec_t));
      ds = (spec_t*) ds_ptr.ptr;

      spec_t* db = NULL;
      DataPtr db_ptr = AllocFromMemoryPool(in_arr->device(), mean_arr->numel() * sizeof(spec_t));
      db = (spec_t*) db_ptr.ptr;

      spec_t* dy_mul_x = NULL;
      DataPtr dy_mul_x_ptr = AllocFromMemoryPool(in_arr->device(), in_arr->numel() * sizeof(spec_t));
      dy_mul_x = (spec_t*) dy_mul_x_ptr.ptr;

      DataPtr gscale_ptr = AllocFromMemoryPool(out_grads->device(), in_arr->numel() * sizeof(spec_t));
      spec_t* gscale = (spec_t*) gscale_ptr.ptr;

      HTShape scale_shape(ndim), scale_stride(ndim);
      int64_t stride_size = 1;
      for (int i = 0; i < ndim; i++) {
        scale_shape[ndim - 1 - i] = (i < reduce_dims ? 
                                     ln_scale->shape(ln_scale->ndim() - 1 - i) : 1);
        scale_stride[ndim - 1 - i] = stride_size;
        stride_size *= scale_shape[ndim - 1 - i];
      }

      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnl::memory::data_type::f32, in_arr->stride());
      auto scale_md = dnnl::memory::desc(scale_shape, dnnl::memory::data_type::f32, scale_stride);
      auto mean_md = dnnl::memory::desc(mean_arr->shape(), dnnl::memory::data_type::f32, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng);
      auto scale_mem = dnnl::memory(scale_md, eng);
      auto mean_mem = dnnl::memory(mean_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(out_grads->data_ptr<spec_t>(), src_mem);
      if (in_arr->shape() == ln_scale->shape())
        hetu::omp::read_from_dnnl_memory(grad_bias->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, scale_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, scale_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(grad_bias->data_ptr<spec_t>(), scale_mem);
      } 

      calculate_gscale<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        gscale, eps, lastdims, (size_t) in_arr->numel());
      
      hetu::omp::write_to_dnnl_memory(gscale, src_mem);
      if (in_arr->shape() == ln_scale->shape())
        hetu::omp::read_from_dnnl_memory(grad_scale->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, scale_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, scale_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(grad_scale->data_ptr<spec_t>(), scale_mem);
      } 

      hetu::omp::write_to_dnnl_memory(out_grads->data_ptr<spec_t>(), src_mem);
      if (in_arr->shape() == mean_arr->shape())
        hetu::omp::read_from_dnnl_memory(db, src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, mean_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, mean_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(db, mean_mem);
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
      // hetu::omp::read_from_dnnl_memory(dy_mul_x, mdst_mem);
      // hetu::omp::write_to_dnnl_memory(dy_mul_x, src_mem);
      if (in_arr->shape() == ln_scale->shape())
        hetu::omp::read_from_dnnl_memory(ds, mdst_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, mean_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, mdst_mem});
        reduction_args.insert({DNNL_ARG_DST, mean_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(ds, mean_mem);
      } 

      calculate_grad_kernel<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ds, db,
        grad_arr->data_ptr<spec_t>(), lastdims, eps, size);

      FreeToMemoryPool(ds_ptr);
      FreeToMemoryPool(db_ptr);
      FreeToMemoryPool(dy_mul_x_ptr);
      FreeToMemoryPool(gscale_ptr);
    }); 
}


} // namespace impl
} // namespace hetu
