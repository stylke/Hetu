#pragma once

#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/dispatch.h"

namespace hetu {
namespace impl {

inline int GetThreadNum(int cnt) {
  if (cnt >= 1048576)
    return 1024;
  if (cnt >= 262144)
    return 512;
  if (cnt >= 65536)
    return 256;
  if (cnt >= 16384)
    return 128;
  if (cnt >= 256)
    return 64;
  return 32;
}

} // namespace impl
} // namespace hetu
