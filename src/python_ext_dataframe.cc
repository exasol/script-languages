#include <Python.h>

#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

struct InputColumnInfo
{
    InputColumnInfo(std::string& name, std::string& type_name) {
        this->name = name;
        this->type_name = type_name;
    }

    std::string name;
    std::string type_name;
};

std::vector<InputColumnInfo> get_column_info(PyObject *exa_meta)
{
//    if (!PyObject_HasAttrString(exa_meta, "input_columns"))
//        return NULL;

    PyObject *py_in_cols = PyObject_GetAttrString(exa_meta, "input_columns");
    if (!py_in_cols)
        throw std::runtime_error("Python exception");
    if (!PyList_Check(py_in_cols))
        throw std::runtime_error("Python exception");

    Py_ssize_t py_num_cols = PyList_Size(py_in_cols);

    std::vector<InputColumnInfo> col_info;

    for (Py_ssize_t i = 0; i < py_num_cols; i++) {
        PyObject *py_col = PyList_GetItem(py_in_cols, i);
        if (!py_col) {
            PyErr_Format(PyExc_IndexError, "Cannot access item %d in exa.meta.input_columns.", i);
            throw std::runtime_error("Python exception");
        }

//        if (!PyObject_HasAttrString(col, "name"))
//            return NULL;

        PyObject *py_col_name = PyObject_GetAttrString(py_col, "name");
        if (!py_col_name)
            throw std::runtime_error("Python exception");

        Py_ssize_t py_col_name_len = 0;
        const char *col_name_buf = PyUnicode_AsUTF8AndSize(py_col_name, &py_col_name_len);
        if (!col_name_buf)
            throw std::runtime_error("Python exception");
        PyObject *py_long_col_name_len = PyLong_FromSsize_t(py_col_name_len);
        if (!py_long_col_name_len)
            throw std::runtime_error("Python exception");
        size_t col_name_len = PyLong_AsSize_t(py_long_col_name_len);
        if (PyErr_Occurred())
            throw std::runtime_error("Python exception");
        std::string col_name(col_name_buf, col_name_len);

        Py_ssize_t py_col_type_name_len = 0;
        PyObject *py_col_type = PyObject_GetAttrString(py_col, "type");
        if (!py_col_type)
            throw std::runtime_error("Python exception");
        PyObject *py_col_type_name = PyObject_GetAttrString(py_col_type, "__name__");
        if (!py_col_type_name)
            throw std::runtime_error("Python exception");
        const char *col_type_name_buf = PyUnicode_AsUTF8AndSize(py_col_type_name, &py_col_type_name_len);
        if (!col_type_name_buf)
            throw std::runtime_error("Python exception");
        PyObject *py_long_col_type_name_len = PyLong_FromSsize_t(py_col_type_name_len);
        if (!py_long_col_type_name_len)
            throw std::runtime_error("Python exception");
        size_t col_type_name_len = PyLong_AsSize_t(py_long_col_type_name_len);
        if (PyErr_Occurred())
            throw std::runtime_error("Python exception");
        std::string col_type_name(col_type_name_buf, col_type_name_len);

        col_info.push_back(InputColumnInfo(col_name, col_type_name));
    }

    return col_info;
}

static PyObject* get_dataframe(PyObject* self, PyObject* args)
{
    PyObject *ctx_iter = NULL;
    PyObject *exa_meta = NULL;
    long num_output_rows = 0;

    if (!PyArg_ParseTuple(args, "OOl", &exa_meta, &ctx_iter, &num_output_rows))
        return NULL;

    try {
        std::vector<InputColumnInfo> in_col_info = get_column_info(exa_meta);
        in_col_info.size();
    }
    catch (std::exception &ex) {
        return NULL;
    }

    if (!PyObject_HasAttrString(ctx_iter, "_exaiter__inp"))
        return NULL;

    PyObject *iter = PyObject_GetAttrString(ctx_iter, "_exaiter__inp");
    if (!iter)
        return NULL;

    Py_XDECREF(iter);

    return Py_BuildValue("s", "Test Test Test");
}

static PyMethodDef dataframe_module_methods[] = {
        {"get_dataframe", get_dataframe, METH_VARARGS},
        {NULL, NULL}
};

static struct PyModuleDef dataframe_module = {
    PyModuleDef_HEAD_INIT,
    "pyextdataframe", /* name of module */
    NULL,
    -1,
    dataframe_module_methods
};

PyMODINIT_FUNC
PyInit_pyextdataframe(void)
{
    return PyModule_Create(&dataframe_module);
}

}
