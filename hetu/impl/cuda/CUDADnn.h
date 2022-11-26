#pragma once

#include "hetu/impl/utils/cuda_utils.h"
#include <cudnn.h>

namespace hetu {
namespace impl {

cudnnHandle_t GetCudnnHandle(int32_t device_id);

} // namespace impl
} // namespace hetu

#define CUDNN_CALL(f)                                                          \
  for (cudnnStatus_t status = (f); status != CUDNN_STATUS_SUCCESS;             \
       status = CUDNN_STATUS_SUCCESS)                                          \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Cudnn call " << #f << " failed: " << cudnnGetErrorString(status)
