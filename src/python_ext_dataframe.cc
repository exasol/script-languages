#include <Python.h>

static PyObject* py_myFunction(PyObject* self, PyObject* args)
{
        char const *s = "Test Test Test";
        return Py_BuildValue("s", s);
}

static PyMethodDef myModule_methods[] = {
        {"myFunction", py_myFunction, METH_VARARGS},
        {NULL, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "pyextdataframe",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    myModule_methods
};

PyMODINIT_FUNC
PyInit_pyextdataframe(void)
{
    return PyModule_Create(&myModule);
}

