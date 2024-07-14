#include "hetu/_binding/graph/recompute.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyPushRecomputeCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_recompute_ctx(List[bool] multi_recompute)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Recompute::push_recompute_enabled(parsed_args.get_bool_list(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyPopRecomputeCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  Recompute::pop_recompute_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyRecomputeCtx_methods[] = { 
  {"push_recompute_ctx", (PyCFunction) PyPushRecomputeCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_recompute_ctx", (PyCFunction) PyPopRecomputeCtx, METH_NOARGS, nullptr},
  {nullptr}
};

void AddRecomputeContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyRecomputeCtx_methods))
    << "Failed to add recompute context managing methods";
}

} // namespace graph
} // namespace hetu
