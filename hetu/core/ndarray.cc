#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include <numeric>
#include <iterator>

namespace hetu {

// A silly serialization method for debugging.
void NDArrayDef::Serialize(std::ostream& os, size_t n_print) const {
  os << "NDArray([";
  size_t size = numel();
  n_print = MIN(n_print, size);
  if (n_print > 0) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      dtype(), spec_t, __FUNCTION__, [&]() {
        if (is_cpu()) {
          const spec_t* ptr = data_ptr<spec_t>();
          os << ptr[0];
          for (size_t i = 1; i < n_print; i++)
            os << ", " << ptr[i];
        } else {
          const spec_t* dev_ptr = data_ptr<spec_t>();
          std::vector<spec_t> host_vec(n_print);
          CudaMemcpy(host_vec.data(), dev_ptr, n_print * sizeof(spec_t),
                     cudaMemcpyDeviceToHost);
          os << host_vec[0];
          for (size_t i = 1; i < n_print; i++)
            os << ", " << host_vec[i];
        }
      });
  }
  os << "], dtype=" << dtype() << ", shape=" << shape()
     << ", device=" << device() << ")";
}

std::ostream& operator<<(std::ostream& os, const NDArray& data) {
  data->Serialize(os);
  return os;
}

NDArray NDArray::EMPTY;
const StreamIndex NDArray::DEFAULT_STREAM = kComputingStream;

NDArray NDArray::to(const NDArray& input, const Device& device, DataType dtype,
                    StreamIndex stream_id, NDArray& output) {
  bool same_device = device.is_undetermined() || device == input->device();
  bool same_dtype = dtype == kUndeterminedDataType || dtype == input->dtype();
  if (same_device && same_dtype) {
    return NDArray(input->meta(), input->storage(), input->storage_offset());
  } else {
    const auto& target_device = same_device ? input->device() : device;
    const auto& target_dtype = same_dtype ? input->dtype() : dtype;
    NDArray out = output.is_defined()
      ? output
      : NDArray::empty(input->shape(), target_device, target_dtype);
    if (output.is_defined()) {
      // Unlike many other kernels, the DataTransfer kernel cannot check
      // whether the devices and dtypes are valid. Hence we check them here.
      HT_ASSERT(output->device() == target_device);
      HT_ASSERT(output->dtype() == target_dtype);
    }
    Stream stream(input->is_cuda() ? input->device() : target_device,
                  stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(stream.device_type(), __FUNCTION__,
                                    hetu::impl::DataTransfer, input, out,
                                    stream);
    return out;
  }
}

NDArray NDArray::abs(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  Stream stream(input->device(), stream_id);
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Abs, input, out, stream);
  return out;
}

NDArray NDArray::add(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::AddElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::add(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::AddConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::sub(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::SubElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::sub(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  return NDArray::add(input, -scalar, stream_id, output);
}

NDArray NDArray::sub(double scalar, const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::SubConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::neg(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Opposite, input, out, stream);
  return out;
}

NDArray NDArray::mul(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::MulElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::mul(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::MulConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::div(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::DivElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::div(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  return NDArray::mul(input, 1.0 / scalar, stream_id, output);
}

NDArray NDArray::div(double scalar, const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::DivConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::pow(const NDArray& input, double exponent,
                     StreamIndex stream_id, NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Pow, input, exponent, out,
                                  stream);
  return out;
}

NDArray NDArray::sqrt(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sqrt, input, out, stream);
  return out;
}

NDArray NDArray::reciprocal(const NDArray& input, StreamIndex stream_id,
                            NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Reciprocal, input, out, stream);
  return out;
}

NDArray NDArray::sigmoid(const NDArray& input, StreamIndex stream_id,
                         NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sigmoid, input, out, stream);
  return out;
}

NDArray NDArray::relu(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Relu, input, out, stream);
  return out;
}

NDArray NDArray::tanh(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Tanh, input, out, stream);
  return out;
}

NDArray NDArray::exp(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Exp, input, out, stream);
  return out;
}

NDArray NDArray::log(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Log, input, out, stream);
  return out;
}

NDArray NDArray::ceil(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Ceil, input, out, stream);
  return out;
}

NDArray NDArray::floor(const NDArray& input, StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Floor, input, out, stream);
  return out;
}

NDArray NDArray::round(const NDArray& input, StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Round, input, out, stream);
  return out;
}

NDArray NDArray::reduce(const NDArray& input, ReductionType red_type,
                        const HTAxes& axes, bool keepdims,
                        StreamIndex stream_id, NDArray& output) {
  auto output_shape = NDArrayMeta::Reduce(input->shape(), axes, keepdims);
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Reduce, input, out, axes, red_type,
                                  stream);
  return out;
}

NDArray NDArray::matmul(const NDArray& x, const NDArray& y, bool trans_left,
                        bool trans_right, StreamIndex stream_id,
                        NDArray& output) {
  HT_ASSERT(x->ndim() == 2 && y->ndim() == 2 &&
            x->shape(trans_left ? 0 : 1) == y->shape(trans_right ? 1 : 0))
    << "Invalid shapes for matrix multiplication: " << x->shape()
    << " (transpose = " << trans_left << ") vs. " << y->shape()
    << " (transpose = " << trans_right << "). ";
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(
        {x->shape(trans_left ? 1 : 0), y->shape(trans_right ? 0 : 1)},
        x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::MatMul, x, trans_left, y,
                                  trans_right, out, stream);
  return out;
}

NDArray NDArray::bmm(const NDArray& x, const NDArray& y,
                     bool trans_left, bool trans_right,
                     StreamIndex stream_id, NDArray& output) {
  const HTShape& a = x->shape();
  const HTShape& b = y->shape();
  int ndims = a.size() - 2;
  HT_ASSERT(a.size() >= 2 && b.size() >= 2 && a.size() == b.size() &&
            a.at(trans_left ? ndims : ndims + 1) ==
              b.at(trans_right ? ndims + 1 : ndims))
    << "Invalid input shapes for:"
    << " (shape_left) " << a << " (shape_right) " << b << " (transpose_left) "
    << trans_left << " (transpose_right) " << trans_right;
  HTShape shape = {};
  for (int i = 0; i < ndims; ++i) {
    HT_ASSERT(a[i] == b[i]);
    shape.emplace_back(a[i]);
  }
  shape.emplace_back(a.at(trans_left ? ndims + 1 : ndims));
  shape.emplace_back(b.at(trans_right ? ndims : ndims + 1));
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::BatchMatMul, x, trans_left, y,
                                  trans_right, out, stream);
  return out;
}

NDArray NDArray::dot(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::MatDot, x, y, out, stream);
  return out;
}

NDArray NDArray::index_add(const NDArray& x, const NDArray& ids,
                           const NDArray& y, int64_t dim, StreamIndex stream_id,
                           NDArray& output) {
  NDArray tmp = NDArray::empty(x->shape(), x->device(), x->dtype());
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(x->shape(), x->device(), x->dtype());
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::IndexAdd, y, ids, tmp, dim,
                                  stream);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::AddElewise, x, tmp, out, stream);
  return out;
}

NDArray NDArray::reshape(const NDArray& input, const HTShape& new_shape,
                         StreamIndex stream_id) {
  // Currently we require storage to be contiguous,
  // so `reshape` will be replaced with `view`.
  (void) stream_id; // suppress un-used warning
  return NDArray::view(input, new_shape);
}

NDArray NDArray::view(const NDArray& input, const HTShape& view_shape) {
  NDArrayMeta output_meta = input->meta();
  output_meta.view(view_shape);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::squeeze(const NDArray& input) {
  NDArrayMeta output_meta = input->meta();
  output_meta.squeeze();
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::squeeze(const NDArray& input, int64_t dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.squeeze(dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::unsqueeze(const NDArray& input, int64_t dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.unsqueeze(dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::flatten(const NDArray& input, int64_t start_dim,
                         int64_t end_dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.flatten(start_dim, end_dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::permute(const NDArray& input, HTAxes& dims,
                         StreamIndex stream_id, NDArray& output) {
  HTShape output_shape = NDArrayMeta::Permute(input->shape(), dims);
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Transpose, input, out,
                                  dims.data(), stream);
  return out;
}

NDArray NDArray::movedim(const NDArray& input, int64_t src, int64_t dst,
                         StreamIndex stream_id, NDArray& output) {
  int64_t len = input->ndim();
  src = NDArrayMeta::ParseAxis(src, len);
  dst = NDArrayMeta::ParseAxis(dst, len);
  HTAxes dims(len);
  if (src < dst) {
    for (int i = 0; i < src; ++i) {
      dims[i] = i;
    }
    for (int i = src; i < dst; ++i) {
      dims[i] = i + 1;
    }
    dims[dst] = src;
    for (int i = dst + 1; i < len; ++i) {
      dims[i] = i;
    }
  } else if (src > dst) {
    for (int i = 0; i < dst; ++i) {
      dims[i] = i;
    }
    dims[dst] = src;
    for (int i = dst + 1; i < src + 1; ++i) {
      dims[i] = i - 1;
    }
    for (int i = src + 1; i < len; ++i) {
      dims[i] = i;
    }
  } else {
    for (int i = 0; i < len; ++i) {
      dims[i] = i;
    }
  }
  return NDArray::permute(input, dims, stream_id, output);
}

NDArray NDArray::adddim(const NDArray& input, int64_t dim, int64_t size,
                        StreamIndex stream_id, NDArray& output) {
  int64_t len = input->ndim();
  dim = NDArrayMeta::ParseAxis(dim, len);
  HT_ASSERT(size > 0);
  HTAxes dims(len);
  HTShape output_shape = {};
  for (int i = 0; i < dim; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  output_shape.emplace_back(size);
  for (int i = dim + 1; i < len; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::BroadcastShape, input, out,
                                  output_shape, stream);
  return out;
}

NDArray NDArray::diagonal(const NDArray& input, int64_t dim1, int64_t dim2,
                          int64_t offset, StreamIndex stream_id,
                          NDArray& output) {
  HTShape output_shape = {};
  int64_t len = input->ndim();
  dim1 = NDArrayMeta::ParseAxis(dim1, len);
  dim2 = NDArrayMeta::ParseAxis(dim2, len);
  for (int i = 0; i < len; ++i) {
    if (i != dim1 && i != dim2) {
      output_shape.emplace_back(input->shape(i));
    }
  }
  output_shape.emplace_back(
    std::min(input->shape(dim1), input->shape(dim2) - offset));
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Diagonal, input, out, dim1, dim2,
                                  offset, stream);
  return out;
}

NDArray NDArray::diagonal_grad(const NDArray& input, int64_t dim1, int64_t dim2,
                               StreamIndex stream_id, NDArray& output) {
  HTShape output_shape = {};
  int64_t len = input->ndim();
  dim1 = NDArrayMeta::ParseAxis(dim1, len + 1);
  dim2 = NDArrayMeta::ParseAxis(dim2, len + 1);
  HT_ASSERT(dim1 < dim2);
  for (int i = 0; i < dim2; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  output_shape.emplace_back(input->shape(dim1));
  for (int i = dim2; i < len; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), input->dtype(),
                                  hetu::impl::DiagonalGradient, input, out,
                                  dim1, dim2, stream);
  return out;
}

NDArrayList NDArray::split(const NDArray& input, size_t num_chunks,
                           int64_t axis, StreamIndex stream_id) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, input->ndim());
  HT_ASSERT(parsed_axis == 0) << "Currently we only support split on axis 0.";
  HT_ASSERT(num_chunks <= static_cast<size_t>(input->shape(parsed_axis)))
    << "Cannot split axis " << axis << " of shape " << input->shape()
    << " into " << num_chunks << " chunks";
  auto avg_chunk_size = DIVUP(input->shape(parsed_axis), num_chunks);
  HTShape chunks(num_chunks, avg_chunk_size);
  chunks[num_chunks - 1] =
    input->shape(parsed_axis) - (num_chunks - 1) * avg_chunk_size;
  return split(input, chunks, axis, stream_id);
}

NDArrayList NDArray::split(const NDArray& input, const HTShape& chunks,
                           int64_t axis, StreamIndex stream_id) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, input->ndim());
  if (parsed_axis == 0) {
    auto split_shapes = NDArrayMeta::Split(input->shape(), chunks, 0);
    size_t interval = input->numel() / input->shape(0);
    NDArrayList ret;
    ret.reserve(split_shapes.size());
    auto offset = input->storage_offset();
    for (size_t i = 0; i < split_shapes.size(); i++) {
      auto split_meta = input->meta();
      split_meta.set_shape(split_shapes[i]);
      ret.emplace_back(split_meta, input->storage(), offset);
      offset += chunks[i] * interval;
    }
    return ret;
  } else {
    HT_NOT_IMPLEMENTED << "Currently we only support split on axis 0.";
    __builtin_unreachable();
  }
}

//_____________________________________________________________________________________

NDArray NDArray::avgpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id,
                         NDArray& output) {  
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t C = input->shape(1);
    int64_t H = input->shape(2);
    int64_t W = input->shape(3);
    int64_t p_H = (H + 2 * padding - kernel_H) / stride + 1;
    int64_t p_W = (W + 2 * padding - kernel_W) / stride + 1;
    out = NDArray::empty({N, C, p_H, p_W}, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::AvgPool,
                                  input, kernel_H, kernel_W,
                                  out, padding, stride,
                                  stream);
  return out;
}

NDArray NDArray::arrayset(NDArray& input, double value,
                          StreamIndex stream_id) {  
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      input->device().type(), __FUNCTION__, hetu::impl::ArraySet, 
      input, value, stream);  
  return input;
}

NDArrayList NDArray::batchnorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias,
                               NDArray& running_mean, NDArray& running_var,
                               double momentum, double eps,
                               StreamIndex stream_id,
                               NDArray& output,
                               NDArray& save_mean,
                               NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  NDArray savemean = save_mean.is_defined() ? save_mean : NDArray::empty({input->shape(1)}, input->device(), input->dtype());
  NDArray savevar = save_var.is_defined() ? save_var : NDArray::empty({input->shape(1)}, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    input->device().type(), __FUNCTION__, hetu::impl::BatchNorm, input,
    bn_scale, bn_bias, out, momentum, eps,
    running_mean, running_var, 
    savemean, savevar, stream);
  return {out, savemean, savevar};
}

NDArray NDArray::broadcast(const NDArray& input, const HTShape& shape,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Broadcast, input,
                                  out, stream);
  return out;
}

NDArray NDArray::broadcast(const NDArray& input, const HTShape& shape,
                           const HTAxes& add_axes,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::BroadcastShape, input,
                                  out, add_axes, stream);
  return out;
}

NDArray NDArray::conv2d(const NDArray& input, const NDArray& filter, 
                        const HTShape& padding, const HTShape& stride,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t H = input->shape(0);
    int64_t W = input->shape(0);
    int64_t f_O = filter->shape(0);
    int64_t f_H = filter->shape(2);
    int64_t f_W = filter->shape(3);
    int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
    int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
    out = NDArray::empty({N, f_O, out_H, out_W}, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Conv2d,
                                  input, filter, out,
                                  padding[0], padding[1],
                                  stride[0], stride[1], stream);
  return out;
}

NDArray NDArray::embedding(const NDArray& input, const NDArray& id,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    HTShape output_shape = id->shape();
    output_shape.emplace_back(input->shape(1));
    out = NDArray::empty(output_shape, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::EmbeddingLookup, input,
                                  id, out, stream);
  return out;
}

NDArray NDArray::gather(const NDArray& input, const NDArray& id, int64_t dim,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(id->shape(), input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Gather, input,
                                  id, out, dim, stream);
  return out;
}

NDArrayList NDArray::instancenorm(const NDArray& input, double eps,
                                  StreamIndex stream_id,
                                  NDArray& output,
                                  NDArray& save_mean,
                                  NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  local_shape[3] = 1;
  local_shape[2] = 1;
  NDArray savemean = save_mean.is_defined() ? save_mean : NDArray::empty(local_shape, input->device(), input->dtype());
  NDArray savevar = save_var.is_defined() ? save_var : NDArray::empty(local_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    input->device().type(), __FUNCTION__, hetu::impl::InstanceNorm, input,
    savemean, savevar, out, eps, stream);
  return {out, savemean, savevar};  
}

NDArray NDArray::kldiv(const NDArray& preds, const NDArray& labels,
                       ReductionType reduction,
                       StreamIndex stream_id,
                       NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype());
    else 
      out = NDArray::empty(preds->shape(), preds->device(), preds->dtype());
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty_like(preds);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::KLDivLoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArrayList NDArray::layernorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias, 
                               const HTShape& normalized_shape, double eps,
                               StreamIndex stream_id,
                               NDArray& output,
                               NDArray& save_mean,
                               NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  NDArray savemean = save_mean.is_defined() ? save_mean : NDArray::empty(normalized_shape, input->device(), input->dtype());
  NDArray savevar = save_var.is_defined() ? save_var : NDArray::empty(normalized_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::LayerNorm, input,
                                  bn_scale, bn_bias, savemean, savevar, 
                                  out, normalized_shape.size(), eps, stream);  
  return {out, savemean, savevar};
}

NDArray NDArray::leakyrelu(const NDArray& input, double alpha,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::LeakyRelu, input,
                                  alpha, out, stream);
  return out;
}

NDArray NDArray::linear(const NDArray& a, const NDArray& b, const NDArray& bias, 
                        bool trans_a, bool trans_b,
                        StreamIndex stream_id,
                        NDArray& output) {
  HT_ASSERT(a->ndim() == 2 && b->ndim() == 2 &&
            a->shape(trans_a ? 0 : 1) == b->shape(trans_b ? 1 : 0))
    << "Invalid shapes for matrix multiplication: " << a->shape()
    << " (transpose = " << trans_a << ") vs. " << b->shape()
    << " (transpose = " << trans_b << "). ";
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(
        {a->shape(trans_a ? 1 : 0), b->shape(trans_b ? 0 : 1)},
        a->device(), a->dtype());
  Stream stream(a->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(a->device().type(), __FUNCTION__, hetu::impl::Linear,
                                  a, trans_a, b, trans_b,
                                  bias, out, stream);
  return out; 
}

NDArray NDArray::mseloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype());
    else 
      out = NDArray::empty(preds->shape(), preds->device(), preds->dtype());
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty_like(preds);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::MSELoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::nllloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype());
    else 
      out = NDArray::empty(labels->shape(), preds->device(), preds->dtype());
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty(labels->shape(), preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::NLLLoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::norm(const NDArray& input, int64_t p, int64_t dim, 
                      bool keepdim,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    HTShape outshape = input->shape();
    int64_t axi = dim >= 0 ? dim: dim + outshape.size();
    if (keepdim) 
      outshape[axi] = 1;
    else 
      outshape.erase(outshape.begin() + axi);
    out = NDArray::empty(outshape, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Norm,
                                  input, out, dim, p, stream);
  return out;
}

NDArray NDArray::onehot(const NDArray& input, size_t num_classes,
                        StreamIndex stream_id,
                        NDArray& output) {
  HTShape out_shape = input->shape();
  out_shape.emplace_back(num_classes);
  NDArray out = output.is_defined() ? output : NDArray::empty(out_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Onehot, input,
                                  num_classes, out, stream);
  return out;
}

NDArray NDArray::pad(const NDArray& input, const HTShape& paddings, 
                     std::string mode, double constant,
                     StreamIndex stream_id,
                     NDArray& output) {
  HTShape Infer = input->shape();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] + paddings[(i - (4 - len / 2)) * 2] +
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  NDArray out = output.is_defined() ? output : NDArray::empty(Infer, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Pad,
                                  input, out, paddings,
                                  stream, mode, constant);
  return out;
}

NDArray NDArray::repeat(const NDArray& input, HTShape repeats,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    HTShape output_shape = repeats;
    HT_ASSERT(output_shape.size() >= input->ndim());
    for (size_t i = 0; i < input->ndim(); ++i) {
      output_shape[i + output_shape.size() - input->ndim()] *= input->shape(i); 
    }
    NDArray::empty(output_shape, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Repeat, input,
                                  out, stream);
  return out;
}

NDArray NDArray::roll(const NDArray& input, HTShape shifts, HTAxes dims,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Roll, input,
                                  shifts, dims,
                                  out, stream);
  return out;
}

NDArray NDArray::sin(const NDArray& input,
                     StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sin, input,
                                  out, stream);
  return out;
}

NDArray NDArray::slice(const NDArray& input, 
                       const HTShape& begin_pos, const HTShape& output_shape,
                       StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(output_shape, input->device(), input->dtype());
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Slice,
                                  input, out,
                                  const_cast<HTShape&>(begin_pos).data(), stream);
  return out;
}

NDArray NDArray::softmax(const NDArray& input,
                         int64_t dim,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Softmax, input,
                                  out, dim, stream);
  return out;
}

NDArray NDArray::sceloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  HTShape output_shape = HTShape(preds->shape().begin(), preds->shape().end() - 1);
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype());
    else 
      out = NDArray::empty(output_shape, preds->device(), preds->dtype());
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty(output_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::SoftmaxCrossEntropy, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::sceloss(const NDArray& preds, const NDArray& labels, const int64_t ignored_index, 
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  HTShape output_shape = HTShape(preds->shape().begin(), preds->shape().end() - 1);
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype());
    else 
      out = NDArray::empty(output_shape, preds->device(), preds->dtype());
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty(output_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::SoftmaxCrossEntropySparse, preds,
                                  labels, unreduced, ignored_index, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}            

NDArray NDArray::triu(const NDArray& input, bool lower, int64_t diagonal, 
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::TriuTril,
                                  input, output, lower, diagonal, stream);
  return out;
}

NDArray NDArray::where(const NDArray& cond, const NDArray& x, const NDArray& y, 
                       StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(x);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__, 
                                  hetu::impl::Where, cond, x, y, out, stream);
  return out;
}

NDArray NDArray::maxpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id,
                         NDArray& output) {  
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t C = input->shape(1);
    int64_t H = input->shape(2);
    int64_t W = input->shape(3);
    int64_t p_H = (H + 2 * padding - kernel_H) / stride + 1;
    int64_t p_W = (W + 2 * padding - kernel_W) / stride + 1;
    out = NDArray::empty({N, C, p_H, p_W}, input->device(), input->dtype());
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::MaxPool,
                                  input, kernel_H, kernel_W,
                                  out, padding, stride,
                                  stream);
  return out;
}

//_____________________________________________________________________________________

NDArray NDArray::cat(const NDArrayList& inputs, int axis,
                     StreamIndex stream_id,
                     NDArray& output) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, inputs.at(0)->ndim());
  if (parsed_axis == 0) {
    std::vector<HTShape> shapes;
    shapes.reserve(inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(shapes),
                   [](const NDArray& x) { return x->shape(); });
    auto cat_shape = NDArrayMeta::Concat(shapes, 0);
    // TODO: For axis 0, we can copy the inputs one-by-one,
    // but it would be better to refine the concat kernel
    // to accept multiple inputs.
    NDArray ret = output.is_defined() ? output :
      NDArray::empty(cat_shape, inputs.at(0)->device(), inputs.at(0)->dtype());
    HTShape chunks(inputs.size());
    std::transform(inputs.begin(), inputs.end(), chunks.begin(),
                   [](const NDArray& x) { return x->shape(0); });
    auto splits = NDArray::split(ret, chunks, 0, stream_id);
    for (size_t i = 0; i < inputs.size(); i++) {
      NDArray::copy(inputs.at(i), stream_id, splits[i]);
    }
    return ret;
  } else {
    HT_NOT_IMPLEMENTED << "Currently we only support concat on axis 0.";
    __builtin_unreachable();
  }
}

NDArray NDArray::empty(const HTShape& shape, const Device& device,
                       DataType dtype) {
  return NDArray(
    NDArrayMeta().set_device(device).set_dtype(dtype).set_shape(shape));
}

NDArray NDArray::empty_like(const NDArray& other) {
  return NDArray::empty(other->shape(), other->device(), other->dtype());
}

NDArray NDArray::full(const HTShape& shape, double fill_value,
                      const Device& device, DataType dtype,
                      StreamIndex stream_id) {
  NDArray out = NDArray::empty(shape, device, dtype);
  return NDArray::full_(out, fill_value, stream_id);
}

NDArray NDArray::full_like(const NDArray& other, double fill_value,
                           StreamIndex stream_id) {
  return NDArray::full(other->shape(), fill_value, other->device(),
                       other->dtype(), stream_id);
}

NDArray NDArray::full_(NDArray& data, double fill_value,
                       StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::ArraySet, data, fill_value,
                                  stream);
  return data;
}

NDArray NDArray::copy(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Device out_device = input->device();
  if (out->device().type() == kCUDA)
    out_device = out->device();
  Stream stream(out_device, stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(out_device.type(), __FUNCTION__,
                                  hetu::impl::DataTransfer, input, out, stream);
  return out;
}

NDArray NDArray::rand(const HTShape& shape, const Device& device,
                      DataType dtype, double lb, double ub, uint64_t seed,
                      StreamIndex stream_id) {
  NDArray out = NDArray::empty(shape, device, dtype);
  return NDArray::uniform_(out, lb, ub, seed, stream_id);
}

NDArray NDArray::randn(const HTShape& shape, const Device& device,
                       DataType dtype, double mean, double stddev,
                       uint64_t seed, StreamIndex stream_id) {
  NDArray out = NDArray::empty(shape, device, dtype);
  return NDArray::normal_(out, mean, stddev, seed, stream_id);
}

NDArray NDArray::uniform_(NDArray& data, double lb, double ub, uint64_t seed,
                          StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::UniformInits, data, lb, ub, seed,
                                  stream);
  return data;
}

NDArray NDArray::normal_(NDArray& data, double mean, double stddev,
                         uint64_t seed, StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::NormalInits, data, mean, stddev,
                                  seed, stream);
  return data;
}

NDArray NDArray::truncated_normal_(NDArray& data, double mean, double stddev,
                                   double lb, double ub, uint64_t seed,
                                   StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::TruncatedNormalInits, data, mean,
                                  stddev, lb, ub, seed, stream);
  return data;
}

} // namespace hetu