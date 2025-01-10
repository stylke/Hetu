#pragma once

#include <Python.h>
#include "hetu/graph/graph.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PySubGraph {
  PyObject_HEAD;
  std::string global_name;
};

extern PyTypeObject* PySubGraph_Type;

inline bool PySubGraph_Check(PyObject* obj) {
  return PySubGraph_Type && PyObject_TypeCheck(obj, PySubGraph_Type);
}

inline bool PySubGraph_CheckExact(PyObject* obj) {
  return PySubGraph_Type && obj->ob_type == PySubGraph_Type;
}

PyObject* PySubGraph_New(std::string subgraph_id);

void AddPySubGraphTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPySubGraph(PyObject* obj) {
  return PySubGraph_Check(obj);
}

inline std::string SubGraphName_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PySubGraph*>(obj)->global_name;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddSubGraphContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hetu
