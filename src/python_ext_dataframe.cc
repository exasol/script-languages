#include <Python.h>

static PyObject* get_dataframe(PyObject* self, PyObject* args)
{
    PyObject *ctx_iter = NULL;
    long num_input_cols = 0;
    long num_output_rows = 0;

    if (!PyArg_ParseTuple(args, "Oll", &ctx_iter, &num_input_cols, &num_output_rows))
        //return "PyArg_ParseTuple failed";
        return NULL;

    if (!PyObject_HasAttrString(ctx_iter, "_exaiter__inp"))
        //return "ctx has no attribute '_exaiter__inp'";
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

