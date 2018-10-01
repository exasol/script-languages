#include <Python.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
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



struct PyPtrDeleter {
    void operator()(PyObject *obj) const noexcept {
        Py_XDECREF(obj);
    }
};

using PyPtr = std::unique_ptr<PyObject, PyPtrDeleter>;

inline void checkPyPtrIsNull(const PyPtr& obj) {
    if (!obj)
        throw std::runtime_error("");
}



struct InputColumnInfo
{
    InputColumnInfo(std::string const& name, std::string const& typeName) {
        this->name = name;
        this->typeName = typeName;
    }

    std::string name;
    std::string typeName;
};



std::vector<InputColumnInfo> getColumnInfo(PyObject *exaMeta)
{
    PyPtr pyInCols(PyObject_GetAttrString(exaMeta, "input_columns"));
    checkPyPtrIsNull(pyInCols);
    if (!PyList_Check(pyInCols.get())) {
        PyErr_SetString(PyExc_RuntimeError, "exa.meta.input_columns is not a list");
        throw std::runtime_error("");
    }

    std::vector<InputColumnInfo> colInfo;

    Py_ssize_t pyNumCols = PyList_Size(pyInCols.get());

    for (Py_ssize_t i = 0; i < pyNumCols; i++) {
        PyPtr pyCol(PyList_GetItem(pyInCols.get(), i));
        checkPyPtrIsNull(pyCol);

        PyPtr pyColName(PyObject_GetAttrString(pyCol.get(), "name"));
        checkPyPtrIsNull(pyColName);
        const char *colName = PyUnicode_AsUTF8(pyColName.get());
        if (!colName)
            throw std::runtime_error("");

        PyPtr pyColType(PyObject_GetAttrString(pyCol.get(), "type"));
        checkPyPtrIsNull(pyColType);
        PyPtr pyColTypeName(PyObject_GetAttrString(pyColType.get(), "__name__"));
        checkPyPtrIsNull(pyColTypeName);
        const char *colTypeName = PyUnicode_AsUTF8(pyColTypeName.get());
        if (!colTypeName)
            throw std::runtime_error("");

        colInfo.push_back(InputColumnInfo(std::string(colName), std::string(colTypeName)));
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
        if (ex.what())
            throw;
        else
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
