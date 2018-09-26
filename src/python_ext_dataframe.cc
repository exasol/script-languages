#include <Python.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

enum ColumnType {
    type_int,
    type_float,
    type_string,
    type_bool,
    type_decimal,
    type_date,
    type_datetime
};

std::map<std::string, ColumnType> column_types {
    {"int", ColumnType::type_int},
    {"float", ColumnType::type_float},
    {"unicode", ColumnType::type_string},
    {"str", ColumnType::type_string},
    {"bool", ColumnType::type_bool},
    {"Decimal", ColumnType::type_decimal},
    {"date", ColumnType::type_date},
    {"datetime", ColumnType::type_datetime}
};

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
    Py_INCREF(exa_meta);

    PyObject *py_in_cols = PyObject_GetAttrString(exa_meta, "input_columns");
    if (!py_in_cols)
        throw std::runtime_error("Python exception");
    if (!PyList_Check(py_in_cols))
        throw std::runtime_error("Python exception");

    std::vector<InputColumnInfo> col_info;

    Py_ssize_t py_num_cols = PyList_Size(py_in_cols);

    for (Py_ssize_t i = 0; i < py_num_cols; i++) {
        PyObject *py_col = PyList_GetItem(py_in_cols, i);
        if (!py_col) {
            PyErr_Format(PyExc_IndexError, "Cannot access item %d in exa.meta.input_columns.", i);
            throw std::runtime_error("Python exception");
        }

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

        Py_XDECREF(py_col_name);

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

        Py_XDECREF(py_col_type);
        Py_XDECREF(py_col_type_name);
    }

    Py_XDECREF(py_in_cols);
    Py_XDECREF(exa_meta);

    return col_info;
}

void get_column_data(std::vector<InputColumnInfo>& col_info, PyObject *ctx_iter, long num_rows)
{
    Py_INCREF(ctx_iter);

    const long num_cols = col_info.size();
    std::vector<ColumnType> colTypes;

    struct PyColumnInfo {
        PyColumnInfo() {
        }
        ~PyColumnInfo() {
            std::vector<PyObject*>::iterator it;
            for (it = pyColumnNums.begin(); it != pyColumnNums.end(); it++) {
                Py_XDECREF(*it);
            }
            for (it = pyMethodNames.begin(); it != pyMethodNames.end(); it++) {
                Py_XDECREF(*it);
            }
        }

        void addColumnNum(PyObject *pyColumnNum) {
            pyColumnNums.push_back(pyColumnNum);
        }
        void addMethodName(PyObject *pyMethodName) {
            pyMethodNames.push_back(pyMethodName);
        }

        PyObject *getColumnNum(long columnNum) {
            return pyColumnNums.at(columnNum);
        }
        PyObject *getMethodName(long columnNum) {
            return pyMethodNames.at(columnNum);
        }

        std::vector<PyObject*> pyColumnNums;
        std::vector<PyObject*> pyMethodNames;
    };

    PyColumnInfo pyColumnInfo;

    for (long i = 0; i < num_cols; i++) {
        colTypes.push_back(column_types[col_info[i].type_name]);

        PyObject *py_col_num = PyLong_FromLong(i);
        if (!py_col_num)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addColumnNum(py_col_num);

        PyObject *py_method_name = NULL;
        switch(colTypes[i]) {
            case ColumnType::type_int:
                py_method_name = PyUnicode_FromString("getInt64");
                break;
            case ColumnType::type_float:
                py_method_name = PyUnicode_FromString("getDouble");
                break;
            case ColumnType::type_string:
                py_method_name = PyUnicode_FromString("getString");
                break;
            case ColumnType::type_bool:
                py_method_name = PyUnicode_FromString("getBoolean");
                break;
            case ColumnType::type_decimal:
                py_method_name = PyUnicode_FromString("getNumeric");
                break;
            case ColumnType::type_date:
                py_method_name = PyUnicode_FromString("getDate");
                break;
            case ColumnType::type_datetime:
                py_method_name = PyUnicode_FromString("getTimestamp");
                break;
            default:
                throw std::runtime_error("Unexpected type");
        }
        if (!py_method_name)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addMethodName(py_method_name);
    }

    for (long r = 0; r < num_rows; r++) {
        for (long c = 0; c < num_cols; c++) {
            PyObject *py_val = PyObject_CallMethodObjArgs(ctx_iter, pyColumnInfo.getMethodName(c), pyColumnInfo.getColumnNum(c), NULL);
            if (!py_val)
                throw std::runtime_error("Python exception");
            Py_XDECREF(py_val);
        }
    }

    Py_XDECREF(ctx_iter);
}

static PyObject* get_dataframe(PyObject* self, PyObject* args)
{
    PyObject *exa_meta = NULL;
    PyObject *ctx_iter = NULL;
    long num_out_rows = 0;

    if (!PyArg_ParseTuple(args, "OOl", &exa_meta, &ctx_iter, &num_out_rows))
        return NULL;

    // Get input column info
    std::vector<InputColumnInfo> in_col_info;
    try {
        in_col_info = get_column_info(exa_meta);
    }
    catch (std::exception &ex) {
        return NULL;
    }

    PyObject *iter = PyObject_GetAttrString(ctx_iter, "_exaiter__inp");
    if (!iter)
        return NULL;

    // Get input data
    try {
        get_column_data(in_col_info, iter, num_out_rows);
    }
    catch (std::exception &ex) {
        Py_XDECREF(iter);
        return NULL;
    }

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
