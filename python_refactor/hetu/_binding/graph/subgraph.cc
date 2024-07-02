#include "hetu/_binding/graph/subgraph.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PySubGraph_New(std::string subgraph_name) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PySubGraph_Type->tp_alloc(PySubGraph_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PySubGraph";
  auto* self = reinterpret_cast<PySubGraph*>(unsafe_self);
  // new(&self->graph_id) SubGraphId();
  self->subgraph_name = subgraph_name;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PySubGraph_dealloc(PyDevice* self) {
  // (&self->graph_id)->~SubGraphId();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PySubGraph_str(PySubGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString("subgraph");
  HT_PY_FUNC_END
}

PyObject* PySubGraph_repr(PySubGraph* self) {
  return PySubGraph_str(self);
}

PyObject* PySubGraph_name(PySubGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->subgraph_name);
  HT_PY_FUNC_END
}

PyObject* PySubGraph_get_subgraph(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_subgraph(std::string subgraph_name)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // Call `SubGraph::GetSubGraph` to check whether `graph_id` is valid
    auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
    return PySubGraph_New(cur_graph.GetSubGraph(parsed_args.get_string(0))->name());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PySubGraph_get_default_subgraph(PyObject*) {
  HT_PY_FUNC_BEGIN
  auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
  return PySubGraph_New(cur_graph.GetSubGraph("global")->name());
  HT_PY_FUNC_END
}

PyObject* PySubGraph_make_new_subgraph(PyObject*, PyObject* args,
                                   PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "make_new_subgraph(std::string subgraph_type=\"\", std::string name=\"\", std::string subgraph_name=\"\")",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
    return PySubGraph_New(
      cur_graph.MakeSubGraph(parsed_args.get_string_or_default(0),
                             parsed_args.get_string_or_default(1),
                             parsed_args.get_string_or_default(2))->name());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PySubGraph_add_op(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "add_op_to_subgraph(Tensor tensor, std::string subgraph_name=\"\")",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // Call `SubGraph::GetSubGraph` to check whether `graph_id` is valid
    auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
    auto tensor = parsed_args.get_tensor(0);
    return PySubGraph_New(cur_graph.AddOpToSubGraph(tensor->producer(), parsed_args.get_string_or_default(1))->name());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPushSubGraphCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_subgraph_ctx(std::string subgraph_name)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
    cur_graph.push_subgraph_ctx(parsed_args.get_string(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPopSubGraphCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "pop_subgraph_ctx()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto& cur_graph = Graph::GetGraph(Graph::cur_graph_ctx());
    cur_graph.pop_subgraph_ctx();
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PySubGraphCtx_methods[] = {
  {"get_subgraph", (PyCFunction) PySubGraph_get_subgraph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"get_default_subgraph", (PyCFunction) PySubGraph_get_default_subgraph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"make_new_subgraph", (PyCFunction) PySubGraph_make_new_subgraph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"push_subgraph_ctx", (PyCFunction) PyPushSubGraphCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_subgraph_ctx", (PyCFunction) PyPopSubGraphCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"add_op_to_subgraph", (PyCFunction) PySubGraph_add_op, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {nullptr}
};

// NOLINTNEXTLINE
PyGetSetDef PySubGraph_properties[] = {
  {PY_GET_SET_DEF_NAME("name"), (getter) PySubGraph_name, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PySubGraph_methods[] = {
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PySubGraph_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.SubGraph", /* tp_name */
  sizeof(PySubGraph), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PySubGraph_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PySubGraph_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PySubGraph_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PySubGraph_methods, /* tp_methods */
  nullptr, /* tp_members */
  PySubGraph_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PySubGraph_Type = &PySubGraph_Type_obj;

void AddPySubGraphTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PySubGraph_Type) < 0) 
    << "PySubGraph_Type not ready";
  Py_INCREF(PySubGraph_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "SubGraph", reinterpret_cast<PyObject*>(PySubGraph_Type)))
    << "Failed to add PySubGraph_Type";
}

void AddSubGraphContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PySubGraphCtx_methods))
    << "Failed to add graph context managing methods";
}

} // namespace graph
} // namespace hetu
