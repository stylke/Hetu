#pragma once

#include "hetu/core/device.h"
#include "hetu/core/dtype.h"

#include <vector>

namespace hetu {

using HTShape = std::vector<int64_t>;
using HTStride = std::vector<int64_t>;
using HTAxes = std::vector<int64_t>;
using HTShapeList = std::vector<HTShape>;
using HTStrideList = std::vector<HTStride>;
using HTKeepDims = std::vector<bool>;

constexpr size_t HT_MAX_NDIM = 10;

namespace {
inline int64_t NumEl(const HTShape& shape) {
  int64_t numel = 1;
  for (auto s : shape)
    numel *= s;
  return numel;
}

inline HTStride Shape2Stride(const HTShape& shape) {
  HTStride stride(shape.size());
  if (shape.size() > 0) {
    stride[shape.size() - 1] = 1;
    for (auto d = shape.size() - 1; d > 0; d--) {
      if (shape[d] == -1 || stride[d] == -1)
        stride[d - 1] = -1;
      else
        stride[d - 1] = stride[d] * shape[d];
    }
  }
  return stride;
}
} // namespace

class NDArrayMeta {
 public:
  NDArrayMeta() = default;
  NDArrayMeta(const HTShape& shape, DataType dtype, const Device& device) {
    set_shape(shape);
    set_dtype(dtype);
    set_device(device);
  }
  NDArrayMeta(const NDArrayMeta&) = default;
  NDArrayMeta(NDArrayMeta&&) = default;
  NDArrayMeta& operator=(NDArrayMeta&&) = default;
  NDArrayMeta& operator=(const NDArrayMeta&) = default;

  inline bool operator==(const NDArrayMeta& meta) const {
    return dtype == meta.dtype && device == meta.device &&
      shape == meta.shape && stride == meta.stride;
  }

  void view(const HTShape& view_shape);

  void unsqueeze(int64_t dim);

  void squeeze();

  void squeeze(int64_t dim);

  void flatten(int64_t start_dim, int64_t end_dim);

  void permute(const HTAxes& axes);

  inline size_t ndim() const {
    return shape.size();
  }

  inline size_t numel() const {
    return NumEl(shape);
  }

  inline NDArrayMeta& set_dtype(DataType t) {
    dtype = t;
    return *this;
  }

  inline NDArrayMeta& set_device(const Device& d) {
    device = d;
    return *this;
  }

  inline NDArrayMeta& set_device(Device&& d) {
    device = std::move(d);
    return *this;
  }

  inline NDArrayMeta& set_shape(const HTShape& s) {
    HT_ASSERT(s.size() <= HT_MAX_NDIM)
      << "Currently we only support up to " << HT_MAX_NDIM
      << " dimensions. Got " << s.size() << ".";
    shape = s;
    stride = Shape2Stride(shape);
    return *this;
  }

  inline NDArrayMeta& set_shape(HTShape&& s) {
    HT_ASSERT(s.size() <= HT_MAX_NDIM)
      << "Currently we only support up to " << HT_MAX_NDIM
      << " dimensions. Got " << s.size() << ".";
    shape = std::move(s);
    stride = Shape2Stride(shape);
    return *this;
  }

  inline NDArrayMeta& set(const NDArrayMeta& other) {
    operator=(other);
    return *this;
  }

  inline NDArrayMeta& set(NDArrayMeta&& other) {
    operator=(std::move(other));
    return *this;
  }

  static HTShape Broadcast(const HTShape& shape_a, const HTShape& shape_b);

  static HTShape Permute(const HTShape& shape, const HTAxes& axes);

  static HTShape Reduce(const HTShape& shape, const HTAxes& axes,
                        bool keepdims);

  static HTShapeList Split(const HTShape& shape, const HTShape& chunks,
                           int axis);

  static HTShape Concat(const HTShapeList& shapes, int64_t axis);

  static int64_t ParseAxis(int64_t axis, int64_t num_dim);

  static HTAxes ParseAxes(const HTAxes axes, int64_t num_dim);

  DataType dtype{kUndeterminedDataType};
  Device device{kUndeterminedDevice};
  HTShape shape;
  HTStride stride;
};

std::ostream& operator<<(std::ostream&, const NDArrayMeta&);

inline bool IsCopiable(const NDArrayMeta& meta1, const NDArrayMeta& meta2) {
  // TODO: support copying between different strides
  return meta1.shape == meta2.shape && meta1.stride == meta2.stride;
}

inline bool IsExchangable(const NDArrayMeta& meta1, const NDArrayMeta& meta2) {
  return meta1.dtype == meta2.dtype && meta1.shape == meta2.shape &&
    meta1.stride == meta2.stride;
}

inline bool IsBroadcastable(const NDArrayMeta& meta1,
                            const NDArrayMeta& meta2) {
  if (meta1.dtype != meta2.dtype)
    return false;
  int32_t len = meta1.shape.size();
  int32_t len2 = meta2.shape.size();
  if (len2 != len + 1)
    return false;
  for (int32_t i = 0; i < len; ++i) {
    if (meta1.shape[i] != meta2.shape[i + 1])
      return false;
  }
  for (int32_t i = 0; i < len; ++i) {
    if (meta1.stride[i] != meta2.stride[i + 1])
      return false;
  }
  return true;
}

inline bool IsConcatable(const NDArrayMeta& meta1, const NDArrayMeta& meta2,
                         const NDArrayMeta& meta3, int64_t axis) {
  if ((meta1.dtype != meta2.dtype) || (meta1.dtype != meta3.dtype) ||
      (meta2.dtype != meta3.dtype))
    return false;
  size_t now_ndim = meta1.ndim();
  if (!(axis >= 0 && (size_t) axis < now_ndim))
    return false;
  if (!(now_ndim == meta2.ndim() && now_ndim == meta3.ndim()))
    return false;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = meta1.shape[i];
    if (!(cur_dim == meta2.shape[i] && cur_dim == meta3.shape[i]))
      return false;
  }
  int64_t offset1 = meta1.shape[axis];
  int64_t offset2 = meta2.shape[axis];
  if (!(offset1 + offset2 == meta3.shape[axis]))
    return false;
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int64_t cur_dim = meta1.shape[i];
    if (!(cur_dim == meta2.shape[i] && cur_dim == meta3.shape[i]))
      return false;
  }
  return true;
}

inline bool IsConcatable(const NDArrayMeta& meta1, const NDArrayMeta& meta2,
                         int64_t axis) {
  if (meta1.dtype != meta2.dtype)
    return false;
  size_t now_ndim = meta1.ndim();
  if (!(axis >= 0 && (size_t) axis < now_ndim))
    return false;
  if (!(now_ndim == meta2.ndim()))
    return false;
  for (int64_t i = 0; i < axis; ++i) {
    int64_t cur_dim = meta1.shape[i];
    if (!(cur_dim == meta2.shape[i]))
      return false;
  }
  for (size_t i = axis + 1; i < now_ndim; i++) {
    int64_t cur_dim = meta1.shape[i];
    if (!(cur_dim == meta2.shape[i]))
      return false;
  }
  return true;
}

} // namespace hetu

namespace std {
inline std::string to_string(const hetu::NDArrayMeta& meta) {
  std::ostringstream os;
  os << meta;
  return os.str();
}
} // namespace std
