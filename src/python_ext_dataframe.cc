#include <Python.h>

static PyObject* get_dataframe(PyObject* self, PyObject* args)
{
    const char *str;
    if (!PyArg_ParseTuple(args, "s", &str))
        return NULL;

    return Py_BuildValue("s", str);
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

