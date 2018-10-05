#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/ufuncobject.h>

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

std::map<std::string, int> typeMap {
    {"bool", NPY_BOOL},
    {"int", NPY_INT32},
    {"intc", NPY_INT32},
    {"intp", NPY_INT64},
    {"int8", NPY_INT8},
    {"int16", NPY_INT16},
    {"int32", NPY_INT32},
    {"int64", NPY_INT64},
    {"uint8", NPY_UINT8},
    {"uint16", NPY_UINT16},
    {"uint32", NPY_UINT32},
    {"uint64", NPY_UINT64},
    {"float", NPY_FLOAT64},
    {"float16", NPY_FLOAT16},
    {"float32", NPY_FLOAT32},
    {"float64", NPY_FLOAT64},
    {"datetime64[ns]", NPY_USERDEF}, // Pandas timestamp: pd.tslib.Timestamp
    {"object", NPY_OBJECT},
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



struct ColumnInfo
{
    ColumnInfo(std::string const& name, std::string const& typeName) {
        this->name = name;
        this->typeName = typeName;
    }

    std::string name;
    std::string typeName;
};



void getColumnInfo(PyObject *exaMeta, const char *columnList, std::vector<ColumnInfo>& colInfo)
{
    PyPtr pyInCols(PyObject_GetAttrString(exaMeta, columnList));
    checkPyPtrIsNull(pyInCols);
    if (!PyList_Check(pyInCols.get())) {
        std::stringstream ss;
        ss << "getColumnInfo: " << columnList << " is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

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

        colInfo.push_back(ColumnInfo(std::string(colName), std::string(colTypeName)));
    }
}

PyObject *getColumnData(std::vector<ColumnInfo>& colInfo, PyObject *ctxIter, long numRows)
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

    PyPtr pyNumRowsLong(PyLong_FromLong(0L));
    checkPyPtrIsNull(pyNumRowsLong);
    Py_ssize_t pyNumRows = PyLong_AsSsize_t(pyNumRowsLong.get());
    if (pyNumRows < 0 && PyErr_Occurred()) 
        throw std::runtime_error("getColumnData(): PyLong_AsSsize_t error");

    PyPtr pyData(PyList_New(pyNumRows));
    for (long r = 0; r < numRows; r++) {
        PyPtr pyRow(PyList_New(numCols));

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

            Py_ssize_t pyColNum = PyLong_AsSsize_t(pyColGetMethods[c].first.get());
            if (pyColNum < 0 && PyErr_Occurred())
                throw std::runtime_error("getColumnData(): PyLong_AsSsize_t error");

            PyObject *item = wasNull ? Py_None : pyVal.release();
            PyList_SET_ITEM(pyRow.get(), pyColNum, item);
        }

        int ok = PyList_Append(pyData.get(), pyRow.get());
        if (ok < 0)
            throw std::runtime_error("getColumnData(): PyList_Append error");

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

    return pyData.release();
}

void emit(PyObject *exaMeta, PyObject *ctxIter, PyObject *dataframe, PyObject *numpyTypes)
{
    PyPtr pd(PyImport_ImportModule("pandas"));
    checkPyPtrIsNull(pd);
    PyPtr np(PyImport_ImportModule("numpy"));
    checkPyPtrIsNull(np);

    // Get output column info
    std::vector<ColumnInfo> outColInfo;
    getColumnInfo(exaMeta, "output_columns", outColInfo);

    std::vector<std::pair<PyPtr, PyPtr>> pyColSetMethods;
    for (unsigned int i = 0; i < outColInfo.size(); i++) {
        PyPtr pyColNum(PyLong_FromLong(i));
        checkPyPtrIsNull(pyColNum);

        ColumnType colType = columnTypes[outColInfo[i].typeName];
        std::string methodName;
        switch(colType) {
            case ColumnType::typeInt:
                methodName = "setInt64";
                break;
            case ColumnType::typeFloat:
                methodName = "setDouble";
                break;
            case ColumnType::typeString:
                methodName = "setString";
                break;
            case ColumnType::typeBoolean:
                methodName = "setBoolean";
                break;
            case ColumnType::typeDecimal:
                methodName = "setNumeric";
                break;
            case ColumnType::typeDate:
                methodName = "setDate";
                break;
            case ColumnType::typeDatetime:
                methodName = "setTimestamp";
                break;
            default: {
                std::stringstream ss;
                ss << "emit(): unexpected type " << colType;
                throw std::runtime_error(ss.str().c_str());
            }
        }
        PyPtr pyMethodName(PyUnicode_FromString(methodName.c_str()));
        checkPyPtrIsNull(pyMethodName);

        pyColSetMethods.push_back(std::make_pair(std::move(pyColNum), std::move(pyMethodName)));
    }

    // Get data type info
    std::vector<int> colTypes;
    PyPtr numpyTypeIter(PyObject_GetIter(numpyTypes));
    checkPyPtrIsNull(numpyTypeIter);
    std::map<std::string, int>::iterator it;
    for (PyPtr numpyType(PyIter_Next(numpyTypeIter.get())); numpyType.get(); numpyType.reset(PyIter_Next(numpyTypeIter.get()))) {
        const char *typeName = PyUnicode_AsUTF8(numpyType.get());
        it = typeMap.find(typeName);
        if (it != typeMap.end()) {
            colTypes.push_back(it->second);
        }
        else {
            std::stringstream ss;
            ss << "emit: unexpected type: " << typeName;
            throw std::runtime_error(ss.str().c_str());
        }
    }

    PyPtr data(PyObject_GetAttrString(dataframe, "values"));
    checkPyPtrIsNull(data);

    PyArrayObject *array = reinterpret_cast<PyArrayObject*>(data.get());
    int numRows = PyArray_DIM(array, 0);
    int numCols = PyArray_DIM(array, 1);

    PyPtr pyNextMethodName(PyUnicode_FromString("next"));
    checkPyPtrIsNull(pyNextMethodName);
    PyPtr pyCheckExceptionMethodName(PyUnicode_FromString("checkException"));
    checkPyPtrIsNull(pyCheckExceptionMethodName);

    PyPtr arrayIter(PyArray_IterNew(data.get()));
    checkPyPtrIsNull(arrayIter);
    PyArrayIterObject *iter = reinterpret_cast<PyArrayIterObject*>(arrayIter.get());
    (void)iter;

    PyPtr pyVal;
    for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
            switch (colTypes[c]) {
                case NPY_INT64:
                case NPY_UINT64:
                {
                    int64_t val = *((int64_t*)(iter->dataptr));
                    PyPtr pyVal(PyLong_FromLong(val));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_INT32:
                case NPY_UINT32:
                {
                    int32_t val = *((int32_t*)(iter->dataptr));
                    PyPtr pyVal(PyLong_FromLong(val));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_INT16:
                case NPY_UINT16:
                {
                    int16_t val = *((int16_t*)(iter->dataptr));
                    PyPtr pyVal(PyLong_FromLong(val));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_INT8:
                case NPY_UINT8:
                {
                    int8_t val = *((int8_t*)(iter->dataptr));
                    PyPtr pyVal(PyLong_FromLong(val));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_FLOAT64:
                {
                    uint64_t val = *((uint64_t*)(iter->dataptr));
                    PyPtr pyVal(PyFloat_FromDouble(static_cast<double>(val)));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_FLOAT32:
                {
                    uint32_t val = *((uint32_t*)(iter->dataptr));
                    PyPtr pyVal(PyFloat_FromDouble(static_cast<double>(val)));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_FLOAT16:
                {
                    uint16_t val = *((uint16_t*)(iter->dataptr));
                    PyPtr pyVal(PyFloat_FromDouble(static_cast<double>(val)));
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_BOOL:
                {
                    bool val = *((bool*)(iter->dataptr));
                    PyPtr pyVal(val ? Py_True : Py_False);
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), NULL));
                    break;
                }
                case NPY_USERDEF: // Pandas timestamp: pd.tslib.Timestamp
                {
                    break;
                }
                case NPY_OBJECT:
                {
                    // Have to check for DATE, NUMERIC, etc.
                    const char *val = (char*)(iter->dataptr);
                    PyPtr pyVal(PyUnicode_FromString(val));
#if 0
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(ctxIter, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), strlen(val), NULL));
#endif
                    break;
                }
                default:
                {
                    std::stringstream ss;
                    ss << "emit: unexpected type: " << colTypes[c];
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            if (!pyVal) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                if (pvalue) {
                    std::stringstream ss;
                    ss << "emit(): Error setting value for row " << r << ", column " << c << ": ";
                    ss << PyUnicode_AsUTF8(pvalue);
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(ctxIter, pyCheckExceptionMethodName.get(), NULL));
            checkPyPtrIsNull(pyCheckException);
            if (pyCheckException.get() != Py_None) {
                const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                if (exMsg) {
                    std::stringstream ss;
                    ss << "emit(): " << exMsg;
                    throw std::runtime_error(ss.str().c_str());
                }
            }
            PyArray_ITER_NEXT(iter);
        }

        PyPtr pyNext(PyObject_CallMethodObjArgs(ctxIter, pyNextMethodName.get(), NULL));
        checkPyPtrIsNull(pyNext);
    }
}

static PyObject *getDataframe(PyObject *self, PyObject *args)
{
    PyObject *exaMeta = NULL;
    PyObject *ctxIter = NULL;
    long numOutRows = 0;

    if (!PyArg_ParseTuple(args, "OOl", &exaMeta, &ctxIter, &numOutRows))
        return NULL;

    PyPtr pyData;
    try {
        PyPtr iter(PyObject_GetAttrString(ctxIter, "_exaiter__inp"));
        checkPyPtrIsNull(iter);
        // Get input column info
        std::vector<ColumnInfo> inColInfo;
        getColumnInfo(exaMeta, "input_columns", inColInfo);
        // Get input data
        pyData.reset(getColumnData(inColInfo, iter.get(), numOutRows));
    }
    catch (std::exception &ex) {
        if (ex.what())
            PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }

    return pyData.release();
}

static PyObject *emitDataframe(PyObject *self, PyObject *args)
{
    PyObject *exaMeta = NULL;
    PyObject *ctxIter = NULL;
    PyObject *dataframe = NULL;
    PyObject *numpyTypes = NULL;

    if (!PyArg_ParseTuple(args, "OOOO", &exaMeta, &ctxIter, &dataframe, &numpyTypes))
        return NULL;

    try {
        PyPtr iter(PyObject_GetAttrString(ctxIter, "_exaiter__out"));
        checkPyPtrIsNull(iter);
        emit(exaMeta, iter.get(), dataframe, numpyTypes);
    }
    catch (std::exception &ex) {
        if (ex.what())
            PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef dataframeModuleMethods[] = {
        {"get_dataframe", getDataframe, METH_VARARGS},
        {"emit_dataframe", emitDataframe, METH_VARARGS},
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
    // Call NumPy import_array() for initialization
    import_array();

    return PyModule_Create(&dataframeModule);
}

}
