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
    if (!PyList_Check(pyInCols.get()))
        throw std::runtime_error("exa.meta.input_columns is not a list");

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
    const long numCols = colInfo.size();
    std::vector<std::pair<PyPtr, PyPtr>> pyColGetMethods;

    for (long i = 0; i < numCols; i++) {
        PyPtr pyColNum(PyLong_FromLong(i));
        checkPyPtrIsNull(pyColNum);

        ColumnType colType = columnTypes[colInfo[i].typeName];
        std::string methodName;
        switch(colType) {
            case ColumnType::typeInt:
                methodName = "getInt64";
                break;
            case ColumnType::typeFloat:
                methodName = "getDouble";
                break;
            case ColumnType::typeString:
                methodName = "getString";
                break;
            case ColumnType::typeBoolean:
                methodName = "getBoolean";
                break;
            case ColumnType::typeDecimal:
                methodName = "getNumeric";
                break;
            case ColumnType::typeDate:
                methodName = "getDate";
                break;
            case ColumnType::typeDatetime:
                methodName = "getTimestamp";
                break;
            default:
                throw std::runtime_error("getColumnData(): unexpected type");
        }
        PyPtr pyMethodName(PyUnicode_FromString(methodName.c_str()));
        checkPyPtrIsNull(pyMethodName);

        pyColGetMethods.push_back(std::make_pair(std::move(pyColNum), std::move(pyMethodName)));
    }

    PyPtr pyWasNullMethodName(PyUnicode_FromString("wasNull"));
    checkPyPtrIsNull(pyWasNullMethodName);
    PyPtr pyNextMethodName(PyUnicode_FromString("next"));
    checkPyPtrIsNull(pyNextMethodName);
    PyPtr pyCheckExceptionMethodName(PyUnicode_FromString("checkException"));
    checkPyPtrIsNull(pyCheckExceptionMethodName);

    for (long r = 0; r < numRows; r++) {
        for (long c = 0; c < numCols; c++) {
            PyPtr pyVal(PyObject_CallMethodObjArgs(ctxIter, pyColGetMethods[c].second.get(), pyColGetMethods[c].first.get(), NULL));
            if (!pyVal) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                std::stringstream ss;
                ss << "getColumnData(): Error fetching value for row " << r << ", column " << c << ": ";
                ss << PyUnicode_AsUTF8(pvalue);
                throw std::runtime_error(ss.str().c_str());
            }

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(ctxIter, pyCheckExceptionMethodName.get(), NULL));
            checkPyPtrIsNull(pyCheckException);
            if (pyCheckException.get() != Py_None) {
                const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                if (exMsg) {
                    std::stringstream ss;
                    ss << "getColumnData(): " << exMsg;
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            PyPtr pyWasNull(PyObject_CallMethodObjArgs(ctxIter, pyWasNullMethodName.get(), NULL));
            checkPyPtrIsNull(pyWasNull);
            int wasNull = PyObject_IsTrue(pyWasNull.get());
            if (wasNull < 0)
                throw std::runtime_error("getColumnData(): wasNull() PyObject_IsTrue() error");
        }

        PyPtr pyNext(PyObject_CallMethodObjArgs(ctxIter, pyNextMethodName.get(), NULL));
        checkPyPtrIsNull(pyNext);

        PyPtr pyCheckException(PyObject_CallMethodObjArgs(ctxIter, pyCheckExceptionMethodName.get(), NULL));
        checkPyPtrIsNull(pyCheckException);
        if (pyCheckException.get() != Py_None) {
            const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
            if (exMsg) {
                std::stringstream ss;
                ss << "getColumnData(): " << exMsg;
                throw std::runtime_error(ss.str().c_str());
            }
        }

        int next = PyObject_IsTrue(pyNext.get());
        if (next < 0)
            throw std::runtime_error("getColumnData(): next() PyObject_IsTrue() error");
        else if (!next)
            break;
    }
}

static PyObject* getDataframe(PyObject* self, PyObject* args)
{
    PyObject *exaMeta = NULL;
    PyObject *ctxIter = NULL;
    long numOutRows = 0;

    if (!PyArg_ParseTuple(args, "OOl", &exaMeta, &ctxIter, &numOutRows))
        return NULL;

    try {
        PyPtr iter(PyObject_GetAttrString(ctxIter, "_exaiter__inp"));
        checkPyPtrIsNull(iter);
        // Get input column info
        std::vector<InputColumnInfo> inColInfo;
        inColInfo = getColumnInfo(exaMeta);
        // Get input data
        getColumnData(inColInfo, iter.get(), numOutRows);
    }
    catch (std::exception &ex) {
        if (ex.what())
            PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }

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
