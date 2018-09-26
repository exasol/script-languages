#include <Python.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

enum ColumnType {
    typeInt,
    typeFloat,
    typeString,
    typeBoolean,
    typeDecimal,
    typeDate,
    typeDatetime
};

std::map<std::string, ColumnType> columnTypes {
    {"int", ColumnType::typeInt},
    {"float", ColumnType::typeFloat},
    {"unicode", ColumnType::typeString},
    {"str", ColumnType::typeString},
    {"bool", ColumnType::typeBoolean},
    {"Decimal", ColumnType::typeDecimal},
    {"date", ColumnType::typeDate},
    {"datetime", ColumnType::typeDatetime}
};

struct InputColumnInfo
{
    InputColumnInfo(std::string& name, std::string& typeName) {
        this->name = name;
        this->typeName = typeName;
    }

    std::string name;
    std::string typeName;
};

std::vector<InputColumnInfo> getColumnInfo(PyObject *exaMeta)
{
    struct PyColumnInfo {
        PyColumnInfo(PyObject *exaMeta) {
            Py_INCREF(exaMeta);
            this->exaMeta = exaMeta;
            this->pyInCols = NULL;
            this->pyColName = NULL;
            this->pyColType = NULL;
            this->pyColTypeName = NULL;
        }
        ~PyColumnInfo() {
            Py_XDECREF(pyInCols);
            Py_XDECREF(pyColName);
            Py_XDECREF(pyColType);
            Py_XDECREF(pyColTypeName);
            Py_XDECREF(exaMeta);
        }

        PyObject *exaMeta;
        PyObject *pyInCols;
        PyObject *pyColName;
        PyObject *pyColType;
        PyObject *pyColTypeName;
    };

    PyColumnInfo pyColInfo(exaMeta);

    pyColInfo.pyInCols = PyObject_GetAttrString(exaMeta, "input_columns");
    if (!pyColInfo.pyInCols)
        throw std::runtime_error("Python exception");
    if (!PyList_Check(pyColInfo.pyInCols))
        throw std::runtime_error("Python exception");

    std::vector<InputColumnInfo> colInfo;

    Py_ssize_t pyNumCols = PyList_Size(pyColInfo.pyInCols);

    for (Py_ssize_t i = 0; i < pyNumCols; i++) {
        PyObject *pyCol = PyList_GetItem(pyColInfo.pyInCols, i);
        if (!pyCol) {
            PyErr_Format(PyExc_IndexError, "Cannot access item %d in exa.meta.input_columns.", i);
            throw std::runtime_error("Python exception");
        }

        pyColInfo.pyColName = PyObject_GetAttrString(pyCol, "name");
        if (!pyColInfo.pyColName)
            throw std::runtime_error("Python exception");

        Py_ssize_t pyColNameLen = 0;
        const char *colNameBuf = PyUnicode_AsUTF8AndSize(pyColInfo.pyColName, &pyColNameLen);
        if (!colNameBuf)
            throw std::runtime_error("Python exception");
        PyObject *pyLongColNameLen = PyLong_FromSsize_t(pyColNameLen);
        if (!pyLongColNameLen)
            throw std::runtime_error("Python exception");
        size_t colNameLen = PyLong_AsSize_t(pyLongColNameLen);
        if (PyErr_Occurred())
            throw std::runtime_error("Python exception");
        std::string colName(colNameBuf, colNameLen);

        Py_ssize_t pyColTypeNameLen = 0;
        pyColInfo.pyColType = PyObject_GetAttrString(pyCol, "type");
        if (!pyColInfo.pyColType)
            throw std::runtime_error("Python exception");
        pyColInfo.pyColTypeName = PyObject_GetAttrString(pyColInfo.pyColType, "__name__");
        if (!pyColInfo.pyColTypeName)
            throw std::runtime_error("Python exception");
        const char *colTypeNameBuf = PyUnicode_AsUTF8AndSize(pyColInfo.pyColTypeName, &pyColTypeNameLen);
        if (!colTypeNameBuf)
            throw std::runtime_error("Python exception");
        PyObject *pyLongColTypeNameLen = PyLong_FromSsize_t(pyColTypeNameLen);
        if (!pyLongColTypeNameLen)
            throw std::runtime_error("Python exception");
        size_t colTypeNameLen = PyLong_AsSize_t(pyLongColTypeNameLen);
        if (PyErr_Occurred())
            throw std::runtime_error("Python exception");
        std::string colTypeName(colTypeNameBuf, colTypeNameLen);

        colInfo.push_back(InputColumnInfo(colName, colTypeName));
    }

    return colInfo;
}

void getColumnData(std::vector<InputColumnInfo>& colInfo, PyObject *ctxIter, long numRows)
{
    struct PyColumnInfo {
        PyColumnInfo(PyObject *ctxIter) {
            Py_INCREF(ctxIter);
            this->ctxIter = ctxIter;
        }
        ~PyColumnInfo() {
            std::vector<PyObject*>::iterator it;
            for (it = pyColumnNums.begin(); it != pyColumnNums.end(); it++) {
                Py_XDECREF(*it);
            }
            for (it = pyMethodNames.begin(); it != pyMethodNames.end(); it++) {
                Py_XDECREF(*it);
            }
            Py_XDECREF(ctxIter);
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

        PyObject *ctxIter;
        std::vector<PyObject*> pyColumnNums;
        std::vector<PyObject*> pyMethodNames;
    };

    PyColumnInfo pyColumnInfo(ctxIter);

    const long numCols = colInfo.size();
    std::vector<ColumnType> colTypes;


    for (long i = 0; i < numCols; i++) {
        colTypes.push_back(columnTypes[colInfo[i].typeName]);

        PyObject *pyColNum = PyLong_FromLong(i);
        if (!pyColNum)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addColumnNum(pyColNum);

        PyObject *pyMethodName = NULL;
        switch(colTypes[i]) {
            case ColumnType::typeInt:
                pyMethodName = PyUnicode_FromString("getInt64");
                break;
            case ColumnType::typeFloat:
                pyMethodName = PyUnicode_FromString("getDouble");
                break;
            case ColumnType::typeString:
                pyMethodName = PyUnicode_FromString("getString");
                break;
            case ColumnType::typeBoolean:
                pyMethodName = PyUnicode_FromString("getBoolean");
                break;
            case ColumnType::typeDecimal:
                pyMethodName = PyUnicode_FromString("getNumeric");
                break;
            case ColumnType::typeDate:
                pyMethodName = PyUnicode_FromString("getDate");
                break;
            case ColumnType::typeDatetime:
                pyMethodName = PyUnicode_FromString("getTimestamp");
                break;
            default:
                throw std::runtime_error("Unexpected type");
        }
        if (!pyMethodName)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addMethodName(pyMethodName);
    }

    for (long r = 0; r < numRows; r++) {
        for (long c = 0; c < numCols; c++) {
            PyObject *pyVal = PyObject_CallMethodObjArgs(ctxIter, pyColumnInfo.getMethodName(c), pyColumnInfo.getColumnNum(c), NULL);
            if (!pyVal)
                throw std::runtime_error("Python exception");
            Py_XDECREF(pyVal);
        }
    }
}

static PyObject* getDataframe(PyObject* self, PyObject* args)
{
    PyObject *exaMeta = NULL;
    PyObject *ctxIter = NULL;
    long numOutRows = 0;

    if (!PyArg_ParseTuple(args, "OOl", &exaMeta, &ctxIter, &numOutRows))
        return NULL;

    // Get input column info
    std::vector<InputColumnInfo> inColInfo;
    try {
        inColInfo = getColumnInfo(exaMeta);
    }
    catch (std::exception &ex) {
        return NULL;
    }

    PyObject *iter = PyObject_GetAttrString(ctxIter, "_exaiter__inp");
    if (!iter)
        return NULL;

    // Get input data
    try {
        getColumnData(inColInfo, iter, numOutRows);
    }
    catch (std::exception &ex) {
        Py_XDECREF(iter);
        return NULL;
    }

    Py_XDECREF(iter);

    return Py_BuildValue("s", "Test Test Test");
}

static PyMethodDef dataframeModuleMethods[] = {
        {"get_dataframe", getDataframe, METH_VARARGS},
        {NULL, NULL}
};

static struct PyModuleDef dataframeModule = {
    PyModuleDef_HEAD_INIT,
    "pyextdataframe", /* name of module */
    NULL,
    -1,
    dataframeModuleMethods
};

PyMODINIT_FUNC
PyInit_pyextdataframe(void)
{
    return PyModule_Create(&dataframeModule);
}

}
