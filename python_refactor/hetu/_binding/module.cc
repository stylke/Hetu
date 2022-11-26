#include "hetu/_binding/module.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/autograd/operator.h"
#include "hetu/_binding/autograd/tensor.h"

PYBIND11_MODULE(HT_CORE_PY_MODULE, m) {
  hetu::AddPyDeviceTypeToModule(m);
  hetu::AddPyDeviceGroupTypeToModule(m);
  hetu::AddPyDataTypeTypeToModule(m);
  hetu::AddPyStreamTypeToModule(m);
  hetu::AddPyNDArrayTypeToModule(m);
  hetu::autograd::AddPyOperatorTypeToModule(m);
  hetu::autograd::AddPyTensorTypeToModule(m);
  hetu::autograd::AddOpContextManagingFunctionsToModule(m);
}
