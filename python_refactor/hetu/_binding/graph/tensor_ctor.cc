#include "hetu/_binding/graph/tensor_ctor.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

#define _PY_TENSOR_CTOR_COMMON_ARGS                                            \
  "DataType dtype=None, bool trainable=false, " OP_META_ARGS

inline PyObject* _from_shape_ctor_helper(ParsedPyArgs& parsed_args, 
                                         Initializer&& init, 
                                         size_t arg_offset = 0) {
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  new(&self->tensor) Tensor();
  self->tensor = MakeVariableOp(
    std::move(init), parsed_args.get_int64_list(0),
    parsed_args.get_dtype_or_peek(arg_offset + 1).value_or(kFloat32), 
    parsed_args.get_bool_or_default(arg_offset + 2), 
    parse_op_meta(parsed_args, arg_offset + 3));
  return reinterpret_cast<PyObject*>(self);
}

inline PyObject* _like_tensor_ctor_helper(ParsedPyArgs& parsed_args,
                                          Initializer&& init, 
                                          size_t arg_offset = 0) {
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  new(&self->tensor) Tensor();
  self->tensor = MakeVariableOp(
    std::move(init), parsed_args.get_tensor(0)->shape(),
    parsed_args.get_dtype_or_peek(arg_offset + 1).value_or(kFloat32),
    parsed_args.get_bool_or_default(arg_offset + 2),
    parse_op_meta(parsed_args, arg_offset + 3));
  return reinterpret_cast<PyObject*>(self);
}

/******************************************************
 * Copy Constrcutor
 ******************************************************/

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);

  static PyArgParser parser({
    "Tensor(numpy.array data, " _PY_TENSOR_CTOR_COMMON_ARGS ")", 
    "Tensor(hetu.NDArray data, " _PY_TENSOR_CTOR_COMMON_ARGS ")", 
    "Tensor(PyObject* data, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  // Question: During device placement, the ndarray data will be copied 
  // if it does not match the expected dtype or device. 
  // Can we defer the copy to device placement?
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromNumpyCtor(
      parsed_args.get_py_obj(0), 
      parsed_args.get_dtype_or_peek(1), 
      nullopt);
    self->tensor = MakeVariableOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parse_op_meta(parsed_args, 3));
  } else if (parsed_args.signature_index() == 1) {
    new (&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromNDArrayCtor(
      parsed_args.get_ndarray(0), parsed_args.get_dtype_or_peek(1), nullopt);
    self->tensor = MakeVariableOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parse_op_meta(parsed_args, 3));
  } else if (parsed_args.signature_index() == 2) {
    new (&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromSequenceCtor(
      parsed_args.get_py_obj(0), parsed_args.get_dtype_or_peek(1), nullopt);
    self->tensor = MakeVariableOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parse_op_meta(parsed_args, 3));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

/******************************************************
 * Placeholder Constrcutor
 ******************************************************/

PyObject* PyTensor_placeholder(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "placeholder(DataType dtype, HTShape shape, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();
    self->tensor =
      MakePlaceholderOp(NDArrayMeta()
                          .set_dtype(parsed_args.get_dtype(0))
                          .set_shape(parsed_args.get_int64_list(1)),
                        parse_op_meta(parsed_args, 2));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  placeholder, 
  (PyCFunction) PyTensor_placeholder, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Empty Tensors
 ******************************************************/

PyObject* PyTensor_empty(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "empty(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, VoidifiedInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  empty, 
  (PyCFunction) PyTensor_empty, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_empty_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "empty_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, VoidifiedInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  empty_like, 
  (PyCFunction) PyTensor_empty_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Zero Tensors
 ******************************************************/

PyObject* PyTensor_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "zeros(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, ZerosInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  zeros, 
  (PyCFunction) PyTensor_zeros, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_zeros_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "zeros_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, ZerosInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  zeros_like, 
  (PyCFunction) PyTensor_zeros_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Ones Tensors
 ******************************************************/

PyObject* PyTensor_ones(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "ones(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, OnesInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  ones, 
  (PyCFunction) PyTensor_ones, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_ones_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "ones_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, OnesInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  ones_like, 
  (PyCFunction) PyTensor_ones_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Full/Filled Tensors
 ******************************************************/

PyObject* PyTensor_full(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "full(HTShape size, double fill_value, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(
      parsed_args, ConstantInitializer(parsed_args.get_float64(1)), 1);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  full, 
  (PyCFunction) PyTensor_full, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_full_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "full_like(Tensor tensor, double fill_value, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(
      parsed_args, ConstantInitializer(parsed_args.get_float64(1)), 1);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  full_like, 
  (PyCFunction) PyTensor_full_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Uniform) Tensors
 ******************************************************/

PyObject* PyTensor_rand(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "rand(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, UniformInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  rand, 
  (PyCFunction) PyTensor_rand, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_rand_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "rand_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, UniformInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  rand_like, 
  (PyCFunction) PyTensor_rand_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Normal) Tensors
 ******************************************************/

PyObject* PyTensor_randn(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "randn(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, NormalInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  randn, 
  (PyCFunction) PyTensor_randn, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_randn_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "randn_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, NormalInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  randn_like, 
  (PyCFunction) PyTensor_randn_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Truncated Normal) Tensors
 ******************************************************/

PyObject* PyTensor_trunc_randn(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "trunc_randn(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, TruncatedNormalInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  trunc_randn, 
  (PyCFunction) PyTensor_trunc_randn, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_trunc_randn_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "trunc_randn_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, TruncatedNormalInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  trunc_randn_like, 
  (PyCFunction) PyTensor_trunc_randn_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

} // namespace graph
} // namespace hetu
