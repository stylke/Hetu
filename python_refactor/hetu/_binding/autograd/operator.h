#pragma once

#include <Python.h>
#include "hetu/autograd/operator.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace autograd {

struct PyOperator {
  PyObject_HEAD;
  Operator op;
};

extern PyTypeObject* PyOperator_Type;

inline bool PyOperator_Check(PyObject* obj) {
  return PyOperator_Type && PyObject_TypeCheck(obj, PyOperator_Type);
}

inline bool PyOperator_CheckExact(PyObject* obj) {
  return PyOperator_Type && obj->ob_type == PyOperator_Type;
}

PyObject* PyOperator_New(const Operator& op,
                         bool return_none_if_undefined = true);

void AddPyOperatorTypeToModule(py::module_& module);

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddOpContextManagingFunctionsToModule(py::module_& module);

ContextManager<TensorList>& get_extra_deps_ctx();

inline OpMeta CurrentOpMetaCtx() {
  OpMeta ret;
  auto stream_index_opt = get_stream_index_ctx().peek();
  if (stream_index_opt != nullopt)
    ret.set_stream_index(*stream_index_opt);
  auto device_group_opt = get_device_group_ctx().peek();
  if (device_group_opt != nullopt)
    ret.set_device_group(*device_group_opt);
  // TODO: name & extra_deps
  return ret;
}

} // namespace autograd
} // namespace hetu
