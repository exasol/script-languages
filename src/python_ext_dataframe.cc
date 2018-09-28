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
            this->wasNullMethodName = NULL;
            this->nextMethodName = NULL;
            this->checkExceptionMethodName = NULL;
        }
        ~PyColumnInfo() {
            std::vector<PyObject*>::iterator it;
            for (it = pyColumnNums.begin(); it != pyColumnNums.end(); it++) {
                Py_XDECREF(*it);
            }
            for (it = pyColumnGetMethodNames.begin(); it != pyColumnGetMethodNames.end(); it++) {
                Py_XDECREF(*it);
            }
            Py_XDECREF(wasNullMethodName);
            Py_XDECREF(nextMethodName);
            Py_XDECREF(checkExceptionMethodName);
            Py_XDECREF(ctxIter);
        }

        void addPyColumnNum(PyObject *pyColumnNum) {
            pyColumnNums.push_back(pyColumnNum);
        }
        void addPyColumnGetMethodName(PyObject *pyMethodName) {
            pyColumnGetMethodNames.push_back(pyMethodName);
        }
        void addColumnGetMethodName(const char *methodName) {
            columnGetMethodNames.push_back(std::string(methodName));
        }

        PyObject *getPyColumnNum(long columnNum) {
            return pyColumnNums.at(columnNum);
        }
        PyObject *getPyColumnGetMethodName(long columnNum) {
            return pyColumnGetMethodNames.at(columnNum);
        }
        const char *getColumnGetMethodName(long columnNum) {
            return columnGetMethodNames.at(columnNum).c_str();
        }

        PyObject *ctxIter;
        PyObject *wasNullMethodName;
        PyObject *nextMethodName;
        PyObject *checkExceptionMethodName;
        std::vector<PyObject*> pyColumnNums;
        std::vector<PyObject*> pyColumnGetMethodNames;
        std::vector<std::string> columnGetMethodNames;
    };

    PyColumnInfo pyColumnInfo(ctxIter);

    const long numCols = colInfo.size();
    std::vector<ColumnType> colTypes;


    for (long i = 0; i < numCols; i++) {
        colTypes.push_back(columnTypes[colInfo[i].typeName]);

        PyObject *pyColNum = PyLong_FromLong(i);
        if (!pyColNum)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addPyColumnNum(pyColNum);

        switch(colTypes[i]) {
            case ColumnType::typeInt:
                pyColumnInfo.addColumnGetMethodName("getInt64");
                break;
            case ColumnType::typeFloat:
                pyColumnInfo.addColumnGetMethodName("getDouble");
                break;
            case ColumnType::typeString:
                pyColumnInfo.addColumnGetMethodName("getString");
                break;
            case ColumnType::typeBoolean:
                pyColumnInfo.addColumnGetMethodName("getBoolean");
                break;
            case ColumnType::typeDecimal:
                pyColumnInfo.addColumnGetMethodName("getNumeric");
                break;
            case ColumnType::typeDate:
                pyColumnInfo.addColumnGetMethodName("getDate");
                break;
            case ColumnType::typeDatetime:
                pyColumnInfo.addColumnGetMethodName("getTimestamp");
                break;
            default:
                throw std::runtime_error("Unexpected type");
        }
        PyObject *pyMethodName = PyUnicode_FromString(pyColumnInfo.getColumnGetMethodName(i));
        if (!pyMethodName)
            throw std::runtime_error("Python exception");
        pyColumnInfo.addPyColumnGetMethodName(pyMethodName);

        pyColumnInfo.wasNullMethodName = PyUnicode_FromString("wasNull");
        if (!pyColumnInfo.wasNullMethodName)
            throw std::runtime_error("Python exception");

        pyColumnInfo.nextMethodName = PyUnicode_FromString("next");
        if (!pyColumnInfo.nextMethodName)
            throw std::runtime_error("Python exception");

        pyColumnInfo.checkExceptionMethodName = PyUnicode_FromString("checkException");
        if (!pyColumnInfo.checkExceptionMethodName)
            throw std::runtime_error("Python exception");
    }

    for (long r = 0; r < numRows; r++) {
        for (long c = 0; c < numCols; c++) {
            PyObject *pyVal = PyObject_CallMethodObjArgs(ctxIter, pyColumnInfo.getPyColumnGetMethodName(c), pyColumnInfo.getPyColumnNum(c), NULL);
            if (!pyVal) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                char *pStrErrorMessage = PyUnicode_AsUTF8(pvalue);
                std::string msg("Col: ");
                msg.append(std::to_string(c));
                msg.append(", Type: ");
                msg.append(Py_TYPE(pyColumnInfo.getPyColumnNum(c))->tp_name);
                msg.append(", ");
                msg.append(pStrErrorMessage);
                throw std::runtime_error(msg.c_str());
            }

            PyObject *pyWasNull = PyObject_CallMethodObjArgs(ctxIter, pyColumnInfo.wasNullMethodName, NULL);
            if (!pyWasNull)
                throw std::runtime_error("wasNull Python exception");
            int wasNull = PyObject_IsTrue(pyWasNull);
            if (wasNull < 0)
                throw std::runtime_error("wasNull isTrue Python exception");
            else if (wasNull)
                throw std::runtime_error("wasNull isTrue");
            Py_XDECREF(pyWasNull);
        }

        PyObject *pyNext = PyObject_CallMethodObjArgs(ctxIter, pyColumnInfo.nextMethodName, NULL);
        if (!pyNext) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            char *pStrErrorMessage = PyUnicode_AsUTF8(pvalue);
            std::string msg("Row: ");
            msg.append(std::to_string(r));
            msg.append(", ");
            msg.append(pStrErrorMessage);
            throw std::runtime_error(msg.c_str());
        }

        PyObject *pyCheckException = PyObject_CallMethodObjArgs(ctxIter, pyColumnInfo.checkExceptionMethodName, NULL);
        if (!pyCheckException)
            throw std::runtime_error("checkException next Python exception");
        if (pyCheckException != Py_None) {
            const char *exMsg = PyUnicode_AsUTF8(pyCheckException);
            if (exMsg) {
                std::string msg("Iterator exception: ");
                msg.append(exMsg);
                throw std::runtime_error(msg.c_str());
            }
        }
        Py_XDECREF(pyCheckException);

        int next = PyObject_IsTrue(pyNext);
        if (next < 0)
            throw std::runtime_error("next isTrue Python exception");
        else if (!next)
            break;
        Py_XDECREF(pyNext);
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
        throw;
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
        throw;
        //return NULL;
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
