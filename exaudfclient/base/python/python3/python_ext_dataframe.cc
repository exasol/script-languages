#include <algorithm>
#include "base/exaudflib/swig/swig_common.h"
#include "base/utils/debug_message.h"

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>

extern "C" {

#define PY_INT (NPY_USERDEF+1)
#define PY_DECIMAL (NPY_USERDEF+2)
#define PY_STR (NPY_USERDEF+3)
#define PY_DATE (NPY_USERDEF+4)
#define PY_NONETYPE (NPY_USERDEF+5)
#define PY_BOOL (NPY_USERDEF+6)
#define PY_FLOAT (NPY_USERDEF+7)
#define PY_TIMESTAMP (NPY_USERDEF+8)

std::map<std::string, int> pandasDTypeStrToNumpyCTypeMap {

    {"int", NPY_INT32},
    {"intc", NPY_INT32},
    {"intp", NPY_INT64},
    {"int8", NPY_INT8},
    {"int16", NPY_INT16},
    {"int32", NPY_INT32},
    {"int64", NPY_INT64},
    {"int8[pyarrow]", NPY_OBJECT},
    {"int16[pyarrow]", NPY_OBJECT},
    {"int32[pyarrow]", NPY_OBJECT},
    {"int64[pyarrow]", NPY_OBJECT},
    
    {"uint8", NPY_UINT8},
    {"uint16", NPY_UINT16},
    {"uint32", NPY_UINT32},
    {"uint64", NPY_UINT64},
    {"uint8[pyarrow]", NPY_OBJECT},
    {"uint16[pyarrow]", NPY_OBJECT},
    {"uint32[pyarrow]", NPY_OBJECT},
    {"uint64[pyarrow]", NPY_OBJECT},
    
    {"float32", NPY_FLOAT32},
    {"float64", NPY_FLOAT64},
    {"float", NPY_FLOAT32},
    {"double", NPY_FLOAT64},
    {"float32[pyarrow]", NPY_OBJECT},
    {"float64[pyarrow]", NPY_OBJECT},
    {"float[pyarrow]", NPY_OBJECT},
    {"double[pyarrow]", NPY_OBJECT},
     // We let numpy convert float16 to float (32 bit) and then use the C conversion from float to double, because a proper conversion from float16 to double in C is very complicated.
    {"float16", NPY_FLOAT32},
    {"halffloat", NPY_FLOAT32},
    {"float16[pyarrow]", NPY_OBJECT},
    {"halffloat[pyarrow]", NPY_OBJECT},

    {"string[pyarrow]", NPY_OBJECT},
    {"string[python]", NPY_OBJECT},
    {"string", NPY_OBJECT},

    {"bool[pyarrow]", NPY_OBJECT},
    {"boolean", NPY_OBJECT},
    {"bool", NPY_BOOL},

    {"datetime64[ns]", NPY_DATETIME},
    {"timestamp[ns, tz=UTC][pyarrow]", NPY_OBJECT},
    
    {"object", NPY_OBJECT},
    
    {"py_NAType", PY_NONETYPE},
    {"py_NoneType", PY_NONETYPE},
    {"py_bool", PY_BOOL},
    {"py_int", PY_INT},
    {"py_float", PY_FLOAT},
    {"py_decimal.Decimal", PY_DECIMAL},
    {"py_str", PY_STR},
    {"py_datetime.date", PY_DATE},
    {"py_Timestamp", PY_TIMESTAMP}
};

std::map<int, std::string> numpyCTypeToNumpyDTypeStrMap {
    {NPY_BOOL, "bool"},
    {NPY_INT8, "int8"},
    {NPY_INT16, "int16"},
    {NPY_INT32, "int32"},
    {NPY_INT64, "int64"},
    {NPY_UINT8, "uint8"},
    {NPY_UINT16, "uint16"},
    {NPY_UINT32, "uint32"},
    {NPY_UINT64, "uint64"},
     // We don't list NPY_FLOAT16 here, because we let numpy convert float16 to float (32 bit) and then use the C conversion from float to double, because a proper conversion from float16 to double in C is very complicated.
    {NPY_FLOAT32, "float32"},
    {NPY_FLOAT64, "float64"},
};

std::map<int, std::string> emitTypeMap {
    {SWIGVMContainers::UNSUPPORTED, "UNSUPPORTED"},
    {SWIGVMContainers::DOUBLE, "DOUBLE"},
    {SWIGVMContainers::INT32, "INT32"},
    {SWIGVMContainers::INT64, "INT64"},
    {SWIGVMContainers::NUMERIC, "NUMERIC"},
    {SWIGVMContainers::TIMESTAMP, "TIMESTAMP"},
    {SWIGVMContainers::DATE, "DATE"},
    {SWIGVMContainers::STRING, "STRING"},
    {SWIGVMContainers::BOOLEAN, "BOOLEAN"},
    {SWIGVMContainers::INTERVALYM, "INTERVALYM"},
    {SWIGVMContainers::INTERVALDS, "INTERVALDS"},
    {SWIGVMContainers::GEOMETRY, "GEOMETRY"}
};



inline void checkPyObjectIsNull(const PyObject *obj, const std::string& error_code) {
    // Error message set by Python
    if (!obj)
        // formerly F-UDF-CL-SL-PYTHON-1038, now we use different error codes 
        // for the calling locations to get better clues what is happing
        throw std::runtime_error(error_code+": PyObject is unexpectedly a null pointer"); 
}


struct PyUniquePtrDeleter {
    void operator()(PyObject *obj) const noexcept {
        Py_XDECREF(obj);
    }
};

using PyUniquePtr = std::unique_ptr<PyObject, PyUniquePtrDeleter>;

struct PyPtr {
    explicit PyPtr() {
    }
    explicit PyPtr(PyObject *obj) {
        checkPyObjectIsNull(obj,"F-UDF-CL-SL-PYTHON-1130");
        ptr.reset(obj);
    }
    void reset(PyObject *obj) {
        ptr.reset(obj);
    }
    PyObject *get() const {
        return ptr.get();
    }
    PyObject *release() {
        return ptr.release();
    }
    explicit operator bool() const {
        return (ptr.get() != nullptr);
    }
    PyUniquePtr ptr;
};

inline void checkPyPtrIsNotNull(const PyPtr& obj) {
    // Error message set by Python
    if (!obj)
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1039");
}

inline void checkPyObjIsNotNull(const PyObject *obj) {
    // Error message set by Python
    if (!obj)
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1142");
}



struct ColumnInfo
{
    ColumnInfo(std::string const& name, long type) {
        this->name = name;
        this->type = static_cast<SWIGVMContainers::SWIGVM_datatype_e>(type);
    }

    std::string name;
    SWIGVMContainers::SWIGVM_datatype_e type;
};



// Global modules
PyPtr datetimeModule(PyImport_ImportModule("datetime"));
PyPtr decimalModule(PyImport_ImportModule("decimal"));
PyPtr pandasModule(PyImport_ImportModule("pandas"));



inline void getColumnInfo(PyObject *ctxIter, PyObject *colNames, long startCol, std::vector<ColumnInfo>& colInfo)
{
    const char *ctxColumnTypeList = "_exaiter__incoltypes";

    PyPtr pyColTypes(PyObject_GetAttrString(ctxIter, ctxColumnTypeList));
    if (!PyList_Check(pyColTypes.get())) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-PYTHON-1040: " << "getColumnInfo: " << ctxColumnTypeList << " is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    if (!PyList_Check(colNames)) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-PYTHON-1041: " << "getColumnInfo: colNames is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    Py_ssize_t pyNumCols = PyList_Size(pyColTypes.get());
    if (pyNumCols != PyList_Size(colNames)) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-PYTHON-1042" << "getColumnInfo: ";
        ss << ctxColumnTypeList << " has length " << pyNumCols << ", but ";
        ss << "colNames has length " << PyList_Size(colNames);
        throw std::runtime_error(ss.str().c_str());
    }

    for (Py_ssize_t i = startCol; i < pyNumCols; i++) {
        PyObject *pyColType = PyList_GetItem(pyColTypes.get(), i);
        checkPyObjectIsNull(pyColType,"F-UDF-CL-SL-PYTHON-1131");
        int colType = PyLong_AsLong(pyColType);
        if (colType < 0 && PyErr_Occurred())
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1043 getColumnInfo(): PyLong_AsLong error");

        PyObject *pyColName = PyList_GetItem(colNames, i);
        checkPyObjectIsNull(pyColName,"F-UDF-CL-SL-PYTHON-1132");
        const char *colName = PyUnicode_AsUTF8(pyColName);
        if (!colName)
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1044");

        colInfo.push_back(ColumnInfo(std::string(colName), colType));
    }
}


inline PyObject *getDateFromString(PyObject *value)
{
    PyPtr datetime(PyObject_GetAttrString(datetimeModule.get(), "datetime"));
    PyPtr pyDatetime(PyObject_CallMethod(datetime.get(), "strptime", "(Os)", value, "%Y-%m-%d"));
    PyPtr pyYear(PyObject_GetAttrString(pyDatetime.get(), "year"));
    PyPtr pyMonth(PyObject_GetAttrString(pyDatetime.get(), "month"));
    PyPtr pyDay(PyObject_GetAttrString(pyDatetime.get(), "day"));
    PyPtr pyDate(PyObject_CallMethod(datetimeModule.get(), "date", "(OOO)", pyYear.get(), pyMonth.get(), pyDay.get()));

    return pyDate.release();
}

inline PyObject *getDatetimeFromString(PyObject *value)
{
    PyPtr datetime(PyObject_GetAttrString(datetimeModule.get(), "datetime"));
    PyPtr pyDatetime(PyObject_CallMethod(datetime.get(), "strptime", "(Os)", value, "%Y-%m-%d %H:%M:%S.%f"));

    return pyDatetime.release();
}

inline PyObject *getLongFromString(PyObject *value)
{
    PyPtr pyLong(PyLong_FromUnicodeObject(value, 10));

    return pyLong.release();
}

inline PyObject *getDecimalFromString(PyObject *value)
{
    PyPtr pyDecimal(PyObject_CallMethod(decimalModule.get(), "Decimal", "(O)", value));

    return pyDecimal.release();
}


PyObject *getColumnData(std::vector<ColumnInfo>& colInfo, PyObject *tableIter, long numRows, long startCol, bool isSetInput, bool& isFinished)
{
    const long numCols = colInfo.size();
    std::vector<std::tuple<Py_ssize_t, PyPtr, PyPtr, std::function<PyObject *(PyObject*)>>> pyColGetMethods;

    for (long i = 0; i < numCols; i++) {
        PyPtr pyColNum(PyLong_FromLong(i + startCol));
        std::function<PyObject *(PyObject*)> postFunction;

        std::string methodName;
        switch(colInfo[i].type) {
            case SWIGVMContainers::INT32:
                methodName = "getInt32";
                break;
            case SWIGVMContainers::INT64:
                methodName = "getInt64";
                break;
            case SWIGVMContainers::DOUBLE:
                methodName = "getDouble";
                break;
            case SWIGVMContainers::NUMERIC:
                postFunction = &getDecimalFromString;
                methodName = "getNumeric";
                break;
            case SWIGVMContainers::STRING:
                methodName = "getString";
                break;
            case SWIGVMContainers::BOOLEAN:
                methodName = "getBoolean";
                break;
            case SWIGVMContainers::DATE:
                postFunction = &getDateFromString;
                methodName = "getDate";
                break;
            case SWIGVMContainers::TIMESTAMP:
                postFunction = &getDatetimeFromString;
                methodName = "getTimestamp";
                break;
            default:
            {
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1045" << "getColumnData(): unexpected type " << colInfo[i].type;
                throw std::runtime_error(ss.str().c_str());
            }
        }
        PyPtr pyMethodName(PyUnicode_FromString(methodName.c_str()));

        Py_ssize_t colNum = PyLong_AsSsize_t(pyColNum.get());
        if (colNum < 0 && PyErr_Occurred())
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1046: getColumnData(): PyLong_AsSsize_t error");

        pyColGetMethods.push_back(std::make_tuple(colNum, std::move(pyColNum), std::move(pyMethodName), postFunction));
    }

    PyPtr pyWasNullMethodName(PyUnicode_FromString("wasNull"));
    PyPtr pyNextMethodName(PyUnicode_FromString("next"));
    PyPtr pyCheckExceptionMethodName(PyUnicode_FromString("checkException"));

    PyPtr pyData(PyList_New(0));
    for (long r = 0; r < numRows; r++) {
        PyPtr pyRow(PyList_New(numCols));

        for (long c = 0; c < numCols; c++) {
            PyPtr pyVal(PyObject_CallMethodObjArgs(tableIter, std::get<2>(pyColGetMethods[c]).get(), std::get<1>(pyColGetMethods[c]).get(), NULL));
            if (!pyVal) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1047: getColumnData(): Error fetching value for row " << r << ", column " << c << ": ";
                ss << PyUnicode_AsUTF8(pvalue);
                throw std::runtime_error(ss.str().c_str());
            }

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(tableIter, pyCheckExceptionMethodName.get(), NULL));
            if (pyCheckException.get() != Py_None) {
                const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                if (exMsg) {
                    std::stringstream ss;
                    ss << "F-UDF-CL-SL-PYTHON-1048: getColumnData(): get row " << r << ", column " << c << ": " << exMsg;
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            PyPtr pyWasNull(PyObject_CallMethodObjArgs(tableIter, pyWasNullMethodName.get(), NULL));
            int wasNull = PyObject_IsTrue(pyWasNull.get());
            if (wasNull < 0)
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1049: getColumnData(): wasNull() PyObject_IsTrue() error");

            if (wasNull) {
                Py_INCREF(Py_None);
                pyVal.reset(Py_None);
            }
            else if (std::get<3>(pyColGetMethods[c])) {
                // Call post function
                pyVal.reset(std::get<3>(pyColGetMethods[c])(pyVal.get()));
            }

            PyList_SET_ITEM(pyRow.get(), std::get<0>(pyColGetMethods[c]) - startCol, pyVal.release());
        }

        int ok = PyList_Append(pyData.get(), pyRow.get());
        if (ok < 0)
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1050: getColumnData(): PyList_Append error");

        if (isSetInput) {
            PyPtr pyNext(PyObject_CallMethodObjArgs(tableIter, pyNextMethodName.get(), NULL));

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(tableIter, pyCheckExceptionMethodName.get(), NULL));
            if (pyCheckException.get() != Py_None) {
                const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                if (exMsg) {
                    std::stringstream ss;
                    ss << "F-UDF-CL-SL-PYTHON-1051: getColumnData(): next(): " << exMsg;
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            int next = PyObject_IsTrue(pyNext.get());
            if (next < 0) {
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1052: getColumnData(): next() PyObject_IsTrue() error");
            }
            else if (!next) {
                isFinished = true;
                break;
            }
        }
    }

    return pyData.release();
}

inline void getColumnSetMethods(std::vector<ColumnInfo>& colInfo, std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods){
    for (unsigned int i = 0; i < colInfo.size(); i++) {
        PyPtr pyColNum(PyLong_FromLong(i));

        std::string methodName;
        switch(colInfo[i].type) {
            case SWIGVMContainers::INT32:
                methodName = "setInt32";
                break;
            case SWIGVMContainers::INT64:
                methodName = "setInt64";
                break;
            case SWIGVMContainers::DOUBLE:
                methodName = "setDouble";
                break;
            case SWIGVMContainers::NUMERIC:
                methodName = "setNumeric";
                break;
            case SWIGVMContainers::STRING:
                methodName = "setString";
                break;
            case SWIGVMContainers::BOOLEAN:
                methodName = "setBoolean";
                break;
            case SWIGVMContainers::DATE:
                methodName = "setDate";
                break;
            case SWIGVMContainers::TIMESTAMP:
                methodName = "setTimestamp";
                break;
            default:
            {
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1053: emit(): unexpected type " << emitTypeMap.at(colInfo[i].type);
                throw std::runtime_error(ss.str().c_str());
            }
        }
        PyPtr pyMethodName(PyUnicode_FromString(methodName.c_str()));

        pyColSetMethods.push_back(std::make_pair(std::move(pyColNum), std::move(pyMethodName)));
    }
}

inline bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.rfind(suffix) == str.size() - suffix.size();
}

inline bool startsWith(std::string const &str, std::string const &prefix) {
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.rfind(prefix, 0) == 0;
}

inline bool isNumpyDatetime64(const char* typeName){
    return startsWith(std::string(typeName),"datetime64[");
}

inline bool isArrowDecimal128(const char* typeName){
    // example decimal128(3, 2)[pyarrow]
    const std::string typeNameStr(typeName);
    return startsWith(typeNameStr, "decimal128(") && endsWith(typeNameStr, "[pyarrow]");
}

inline void getColumnTypeInfo(PyObject *numpyTypes, std::vector<std::pair<std::string, int>>& colTypes){
    PyPtr numpyTypeIter(PyObject_GetIter(numpyTypes));
    for (PyPtr numpyType(PyIter_Next(numpyTypeIter.get())); numpyType.get(); numpyType.reset(PyIter_Next(numpyTypeIter.get()))) {
        const char *typeName = PyUnicode_AsUTF8(numpyType.get());
        std::map<std::string, int>::iterator it = pandasDTypeStrToNumpyCTypeMap.find(typeName);
        if (it != pandasDTypeStrToNumpyCTypeMap.end()) {
            colTypes.push_back(*it);
        } else if(isArrowDecimal128(typeName)){
            colTypes.push_back({typeName, NPY_OBJECT});
        } else if(isNumpyDatetime64(typeName)){
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1138: emit: unsupported datetime type: " << typeName << 
                " emit dataframe only supports datetime64[ns] which represents datetime with nanosecond resolution without time zone. " <<
                "If you need time zone please use an additional result column. " << 
                "Note: the nanosecond resolution gets truncated by the database to milliseconds";
            throw std::runtime_error(ss.str().c_str());
        } else {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1054: emit: unexpected type: " << typeName;
            throw std::runtime_error(ss.str().c_str());
        }
    }

}

inline void printPyObject(PyObject* obj, const std::string& error_code){
        PyTypeObject* type = obj->ob_type;
        const char* p = type->tp_name;
        PyObject* objectsRepresentation = PyObject_Repr(obj);
        const char* s =  PyUnicode_AsUTF8(objectsRepresentation);
        DBG_STREAM_MSG(std::cerr, error_code << ": " << std::string(s) << " " << std::string(p));
}

inline const PyPtr& getPandasNA(){
    static const PyPtr pdNA(PyObject_GetAttrString(pandasModule.get(), "NA"));
    return pdNA;
}

inline bool isNoneOrNA(PyObject* pyVal){
    const PyPtr& pdNA = getPandasNA();
    return pyVal == Py_None  || pyVal == pdNA.get();
}

inline void getColumnArrays(PyObject *colArray, int numCols, int numRows, 
        std::vector<std::pair<std::string, int>>& colTypes, std::vector<PyPtr>& columnArrays){
    for (int c = 0; c < numCols; c++) {
        PyPtr pyStart(PyLong_FromLong(c));
        PyPtr pyStop(PyLong_FromLong(c + 1));
        PyPtr slice(PySlice_New(pyStart.get(), pyStop.get(), Py_None));
        PyPtr arraySlice(PyObject_GetItem(colArray, slice.get()));

        PyPtr pyZero(PyLong_FromLong(0L));
        PyPtr array(PyObject_GetItem(arraySlice.get(), pyZero.get()));


        if (colTypes[c].second == NPY_OBJECT) {
            // Convert numpy array to python list
            PyPtr pyList(PyObject_CallMethod(array.get(), "tolist", NULL));
            if (!PyList_Check(pyList.get())) {
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1055: emit(): column array " << c << " is not a list";
                throw std::runtime_error(ss.str().c_str());
            }

            // Get type of first non-None, non-NA item in list
            PyObject *pyVal = PyList_GetItem(pyList.get(), 0);
            checkPyObjectIsNull(pyVal,"F-UDF-CL-SL-PYTHON-1126");
            std::string pyTypeName(std::string("py_") + Py_TYPE(pyVal)->tp_name);
            bool pyValIsNoneOrNA = isNoneOrNA(pyVal);
            for (int r = 1; r < numRows && pyValIsNoneOrNA; r++) {
                pyVal = PyList_GetItem(pyList.get(), r);
                pyValIsNoneOrNA = isNoneOrNA(pyVal);
                checkPyObjectIsNull(pyVal,"F-UDF-CL-SL-PYTHON-1127");
                if (!pyValIsNoneOrNA) {
                    pyTypeName = std::string("py_") + Py_TYPE(pyVal)->tp_name;
                    break;
                }
            }

            // Update type in column type info
            std::map<std::string, int>::iterator userDefIt;
            userDefIt = pandasDTypeStrToNumpyCTypeMap.find(pyTypeName);
            if (userDefIt != pandasDTypeStrToNumpyCTypeMap.end()) {
                colTypes[c] = *userDefIt;
            } else {
                // TODO accept pandas.Timestamp values
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1056: emit: column " <<  c << ", unexpected python type: " << pyTypeName;
                throw std::runtime_error(ss.str().c_str());
            }

            columnArrays.push_back(std::move(pyList));
        }
        else if (colTypes[c].second == NPY_DATETIME) {
            
            
            // Convert numpy array to python list
            PyPtr pyList(PyObject_CallMethod(array.get(), "tolist", NULL));

            if (!PyList_Check(pyList.get())) {
                std::stringstream ss;
                ss << "F-UDF-CL-SL-PYTHON-1057: emit(): column array " << c << " is not a list";
                throw std::runtime_error(ss.str().c_str());
            }

            columnArrays.push_back(std::move(pyList));
        }
        else {
            PyPtr asType (PyObject_GetAttrString(array.get(), "astype"));
            PyPtr keywordArgs(PyDict_New());
            PyDict_SetItemString(keywordArgs.get(), "copy", Py_False);
            const std::string numpyDTypeStr = numpyCTypeToNumpyDTypeStrMap.at(colTypes[c].second);
            PyPtr funcArgs(Py_BuildValue("(s)", numpyDTypeStr.c_str()));
            PyPtr scalarArr(PyObject_Call(asType.get(), funcArgs.get(), keywordArgs.get()));
            columnArrays.push_back(std::move(scalarArr));
        }
    }

}

inline void handleEmitNpyUint64(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    int64_t value = *((int64_t*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(value));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(static_cast<double>(value)));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1058: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyUint32(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    int32_t value = *((int32_t*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(value));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(static_cast<double>(value)));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1059: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyUint16(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    int16_t value = *((int16_t*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(value));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(static_cast<double>(value)));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1060: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyUint8(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    int8_t value = *((int8_t*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(value));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(static_cast<double>(value)));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1061: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyFloat64(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    double value = *((double*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(static_cast<int64_t>(value)));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(value));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1062: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyFloat32(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    double value = *((float*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
            pyValue.reset(PyLong_FromLong(static_cast<int64_t>(value)));
            break;
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyUnicode_FromString(std::to_string(value).c_str()));
            break;
        case SWIGVMContainers::DOUBLE:
            pyValue.reset(PyFloat_FromDouble(value));
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1063: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitNpyBool(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    bool value = *((bool*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
    if (npy_isnan(value)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::BOOLEAN:
            if (value) {
                Py_INCREF(Py_True);
                pyValue.reset(Py_True);
            }
            else {
                Py_INCREF(Py_False);
                pyValue.reset(Py_False);
            }
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1065: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitPyBool(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyBool = PyList_GetItem(columnArrays[c].get(), r);
    checkPyObjIsNotNull(pyBool);
    if (isNoneOrNA(pyBool)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }
    switch (colInfo[c].type) {
        case SWIGVMContainers::BOOLEAN:
            if (pyBool == Py_True) {
                Py_INCREF(Py_True);
                pyValue.reset(Py_True);
            }
            else if (pyBool == Py_False) {
                Py_INCREF(Py_False);
                pyValue.reset(Py_False);
            }
            else {
                pyValue.reset(nullptr);
            }
            break;
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1066: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    checkPyPtrIsNotNull(pyValue);
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitPyInt(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyInt = PyList_GetItem(columnArrays[c].get(), r);
    checkPyObjIsNotNull(pyInt);
    if (isNoneOrNA(pyInt)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
        {
            //pyInt points to a 'borrowed' reference. We need to explicitly increase the ref counter here, as pyValue will decrease it again later.
            Py_INCREF(pyInt);
            pyValue.reset(pyInt);
            break;
        }
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyObject_Str(pyInt));
            break;
        case SWIGVMContainers::DOUBLE:
        {
            double value = PyFloat_AsDouble(pyInt);
            if (value < 0 && PyErr_Occurred())
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1067: emit() PY_INT: PyFloat_AsDouble error");
            pyValue.reset(PyFloat_FromDouble(value));
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1068: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitPyFloat(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
     //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyFloat = PyList_GetItem(columnArrays[c].get(), r);
    checkPyObjIsNotNull(pyFloat);
    if (isNoneOrNA(pyFloat)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
        {
            double value = PyFloat_AsDouble(pyFloat);
            if (value < 0 && PyErr_Occurred())
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1139: emit() PY_FLOAT: PyFloat_AsDouble error");
            if (npy_isnan(value)) {
                pyResult.reset(
                  PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
                return;
            } 
            pyValue.reset(PyLong_FromLong(static_cast<int64_t>(value)));
            break;
        }
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyObject_Str(pyFloat));
            break;
        case SWIGVMContainers::DOUBLE:
        {
            //pyFloat points to a 'borrowed' reference. We need to explicitly increase the ref counter here, as pyValue will decrease it again later.
            Py_INCREF(pyFloat);
            pyValue.reset(pyFloat);
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1140: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitPyDecimal(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName,
        PyPtr& pyIntMethodName,
        PyPtr& pyFloatMethodName
        ){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyDecimal = PyList_GetItem(columnArrays[c].get(), r);
    if (isNoneOrNA(pyDecimal)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::INT64:
        case SWIGVMContainers::INT32:
        {
            PyPtr pyInt(PyObject_CallMethodObjArgs(pyDecimal, pyIntMethodName.get(), NULL));
            pyValue.reset(pyInt.release());
            break;
        }
        case SWIGVMContainers::NUMERIC:
            pyValue.reset(PyObject_Str(pyDecimal));
            break;
        case SWIGVMContainers::DOUBLE:
        {
            PyPtr pyFloat(PyObject_CallMethodObjArgs(pyDecimal, pyFloatMethodName.get(), NULL));
            pyValue.reset(pyFloat.release());
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1069: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
}

inline void handleEmitPyStr(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyString = PyList_GetItem(columnArrays[c].get(), r);
    if (isNoneOrNA(pyString)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::NUMERIC:
            pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyString, NULL));
            break;
        case SWIGVMContainers::STRING:
        {
            Py_ssize_t size = -1;
            const char *str = PyUnicode_AsUTF8AndSize(pyString, &size);
            if (!str && size < 0)
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1137: invalid size of string");
            PyPtr pySize(PyLong_FromSsize_t(size));
            pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyString, pySize.get(), NULL));
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1070: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

inline void handleEmitPyDate(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName,
        PyPtr& pyIsoformatMethodName){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyDate = PyList_GetItem(columnArrays[c].get(), r);
    if (isNoneOrNA(pyDate)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::DATE:
        {
            PyPtr pyIsoDate(PyObject_CallMethodObjArgs(pyDate, pyIsoformatMethodName.get(), NULL));
            pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyIsoDate.get(), NULL));
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1071: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
}
inline void handleEmitPyTimestamp(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyTimestamp = PyList_GetItem(columnArrays[c].get(), r);
    if (isNoneOrNA(pyTimestamp)) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::TIMESTAMP:
        {
            // We call here pandas.Timestamp.tz_localize(None), because we need to remove the timezone from the timestamp.
            // Exasol doesn't support timezones, and if we don't remove the timezone, pandas.Timestamp.isoformat will add
            // it to the generated string.
            PyPtr pyTzLocalize(PyObject_CallMethod(pyTimestamp, "tz_localize", "z", NULL));
            PyPtr pyIsoDatetime(PyObject_CallMethod(pyTzLocalize.get(), "isoformat", "s", " "));
            pyResult.reset(PyObject_CallMethodObjArgs(
                resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyIsoDatetime.get(), NULL));
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1141: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
}


inline void handleEmitNpyDateTime(
        int c, int r,
        std::vector<PyPtr>& columnArrays,
        std::vector<std::pair<PyPtr, PyPtr>>& pyColSetMethods,
        std::vector<ColumnInfo>& colInfo,
        std::vector<std::pair<std::string, int>>& colTypes,
        PyObject *resultHandler,
        PyPtr& pyValue,
        PyPtr& pyResult,
        PyPtr& pySetNullMethodName,
        PyPtr& pdNaT){
    //PyList_GetItem returns a 'borrowed' reference. Must not call XDECREF() on it.
    PyObject *pyTimestamp = PyList_GetItem(columnArrays[c].get(), r);
    if (pyTimestamp == pdNaT.get()) {
        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
        return;
    }

    switch (colInfo[c].type) {
        case SWIGVMContainers::TIMESTAMP:
        {
            PyPtr pyIsoDatetime(PyObject_CallMethod(pyTimestamp, "isoformat", "s", " "));
            pyResult.reset(PyObject_CallMethodObjArgs(
                resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyIsoDatetime.get(), NULL));
            break;
        }
        default:
        {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-PYTHON-1072: emit column " << c << " of type " << emitTypeMap.at(colInfo[c].type) << " but data given have type " << colTypes[c].first;
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

void emit(PyObject *resultHandler, std::vector<ColumnInfo>& colInfo, PyObject *dataframe, PyObject *numpyTypes)
{
    std::vector<std::pair<PyPtr, PyPtr>> pyColSetMethods;
    try{
        getColumnSetMethods(colInfo, pyColSetMethods);
    } catch (std::exception& err){
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1133: "+std::string(err.what()));
    }
    
    std::vector<std::pair<std::string, int>> colTypes;
    try{
        getColumnTypeInfo(numpyTypes, colTypes);
    } catch (std::exception& err){
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1134: "+std::string(err.what()));
    }

    int numRows = -1;
    int numCols = -1;


    PyPtr data;
    PyPtr arrayPtr;
    PyArrayObject *pyArray;
    PyPtr colArray;
    bool allColsAreDateTime =
        std::all_of(colTypes.begin(), colTypes.end(),
                    [](std::pair<std::string, int> colType) {
                        return colType.second == NPY_DATETIME;
                    });
    if(allColsAreDateTime) {
        // if we get an dataframe with only datetime columns with type datetime[ns],
        // As we call PyArray_FROM_OTF(data.get(), NPY_OBJECT, NPY_ARRAY_IN_ARRAY) with parameter NPY_OBJECT we need
        // to explicitly cast the values to type 'object'. Per default the dtypes of values are datetime[ns].
        PyPtr asTypeFunc (PyObject_GetAttrString(dataframe, "astype"));
        PyPtr keywordArgs(PyDict_New());
        PyDict_SetItemString(keywordArgs.get(), "copy", Py_False);
        PyPtr funcArgs(Py_BuildValue("(s)", "object"));
        PyPtr castedValues(PyObject_Call(asTypeFunc.get(), funcArgs.get(), keywordArgs.get()));
        data.reset(PyObject_GetAttrString(castedValues.get(), "values"));
        arrayPtr = PyPtr(PyArray_FROM_OTF(data.get(), NPY_OBJECT, NPY_ARRAY_IN_ARRAY));
        pyArray = reinterpret_cast<PyArrayObject*>(arrayPtr.get());
        numRows = PyArray_DIM(pyArray, 0);
        numCols = PyArray_DIM(pyArray, 1);
        // Transpose to column-major
        colArray = PyPtr(PyArray_Transpose(pyArray, NULL));
    }else{
        data=PyPtr(PyObject_GetAttrString(dataframe, "values"));
        arrayPtr = PyPtr(PyArray_FROM_OTF(data.get(), NPY_OBJECT, NPY_ARRAY_IN_ARRAY));
        pyArray = reinterpret_cast<PyArrayObject*>(arrayPtr.get());
        numRows = PyArray_DIM(pyArray, 0);
        numCols = PyArray_DIM(pyArray, 1);
        // Transpose to column-major
        colArray = PyPtr(PyArray_Transpose(pyArray, NULL));
    }

    // Get column arrays
    std::vector<PyPtr> columnArrays;
    try{
        getColumnArrays(colArray.get(), numCols, numRows, colTypes, columnArrays);
    } catch (std::exception& err){
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1135: "+std::string(err.what()));
    }

    try{
        PyPtr pySetNullMethodName(PyUnicode_FromString("setNull"));
        PyPtr pyNextMethodName(PyUnicode_FromString("next"));
        PyPtr pyCheckExceptionMethodName(PyUnicode_FromString("checkException"));
        PyPtr pyIntMethodName(PyUnicode_FromString("__int__"));
        PyPtr pyFloatMethodName(PyUnicode_FromString("__float__"));
        PyPtr pyIsoformatMethodName(PyUnicode_FromString("isoformat"));
        PyPtr pdNaT(PyObject_GetAttrString(pandasModule.get(), "NaT"));

        // Emit data
        PyPtr pyValue;
        PyPtr pyResult;
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                switch (colTypes[c].second) {
                    case NPY_INT64:
                    case NPY_UINT64:
                    {
                        handleEmitNpyUint64(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_INT32:
                    case NPY_UINT32:
                    {
                        handleEmitNpyUint32(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_INT16:
                    case NPY_UINT16:
                    {
                        handleEmitNpyUint16(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_INT8:
                    case NPY_UINT8:
                    {   
                        handleEmitNpyUint8(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_FLOAT64:
                    {   
                        handleEmitNpyFloat64(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_FLOAT32:
                    {
                        handleEmitNpyFloat32(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case NPY_BOOL:
                    {
                        handleEmitNpyBool(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case PY_BOOL:
                    {
                        handleEmitPyBool(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case PY_INT:
                    {
                        handleEmitPyInt(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case PY_FLOAT:
                    {
                        handleEmitPyFloat(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case PY_DECIMAL:
                    {
                        handleEmitPyDecimal(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, 
                                            pySetNullMethodName, pyIntMethodName, pyFloatMethodName);
                        break;
                    }
                    case PY_STR:
                    {
                        handleEmitPyStr(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, pySetNullMethodName);
                        break;
                    }
                    case PY_DATE:
                    {
                        handleEmitPyDate(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, 
                                        pySetNullMethodName, pyIsoformatMethodName);
                        break;
                    }
                    case PY_TIMESTAMP:
                    {
                        handleEmitPyTimestamp(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, 
                                        pySetNullMethodName);
                        break;
                    }
                    case NPY_DATETIME:
                    {
                        handleEmitNpyDateTime(c, r, columnArrays, pyColSetMethods, colInfo, colTypes, resultHandler, pyValue, pyResult, 
                                            pySetNullMethodName, pdNaT);
                        break;
                    }
                    case PY_NONETYPE:
                    {
                        pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pySetNullMethodName.get(), pyColSetMethods[c].first.get(), NULL));
                        break;
                    }
                    default:
                    {
                        std::stringstream ss;
                        ss << "F-UDF-CL-SL-PYTHON-1073: emit: unexpected type: " << colTypes[c].first;
                        throw std::runtime_error(ss.str().c_str());
                    }
                }

                if (!pyResult) {
                    PyObject *ptype, *pvalue, *ptraceback;
                    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                    if (pvalue) {
                        std::stringstream ss;
                        ss << "F-UDF-CL-SL-PYTHON-1074: emit(): Error setting value for row " << r << ", column " << c << ": ";
                        ss << PyUnicode_AsUTF8(pvalue);
                        throw std::runtime_error(ss.str().c_str());
                    }
                }

                PyPtr pyCheckException(PyObject_CallMethodObjArgs(resultHandler, pyCheckExceptionMethodName.get(), NULL));
                if (pyCheckException.get() != Py_None) {
                    const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                    if (exMsg) {
                        std::stringstream ss;
                        ss << "F-UDF-CL-SL-PYTHON-1075: emit(): " << exMsg;
                        throw std::runtime_error(ss.str().c_str());
                    }
                }
            }

            PyPtr pyNext(PyObject_CallMethodObjArgs(resultHandler, pyNextMethodName.get(), NULL));
        }
    }catch (std::exception& err){
        throw std::runtime_error("F-UDF-CL-SL-PYTHON-1136: "+std::string(err.what()));
    }
}

PyObject *createDataFrame(PyObject *data, std::vector<ColumnInfo>& colInfo)
{
    PyPtr pdDataFrame(PyObject_GetAttrString(pandasModule.get(), "DataFrame"));

    Py_ssize_t numCols = static_cast<Py_ssize_t>(colInfo.size());
    PyPtr pyColumnNames(PyList_New(numCols));
    for (Py_ssize_t i = 0; i < numCols; i++) {
        PyPtr pyColName(PyUnicode_FromString(colInfo[i].name.c_str()));
        PyList_SET_ITEM(pyColumnNames.get(), i, pyColName.release());
    }

    PyPtr funcArgs(Py_BuildValue("(O)", data));

    PyPtr keywordArgs(PyDict_New());
    PyDict_SetItemString(keywordArgs.get(), "columns", pyColumnNames.get());

    PyPtr pyDataFrame(PyObject_Call(pdDataFrame.get(), funcArgs.get(), keywordArgs.get()));

    return pyDataFrame.release();
}

PyObject *getNumpyTypes(PyObject *dataframe)
{
    PyPtr pyDtypes(PyObject_GetAttrString(dataframe, "dtypes"));
    PyPtr pyDtypeValues(PyObject_CallMethod(pyDtypes.get(), "tolist", NULL));

    if (!PyList_Check(pyDtypeValues.get())) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-PYTHON-1076: DataFrame.dtypes.values is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    Py_ssize_t pyNumCols = PyList_Size(pyDtypeValues.get());
    PyPtr pyColumnDtypes(PyList_New(pyNumCols));
    for (Py_ssize_t i = 0; i < pyNumCols; i++) {
        PyObject *pyColDtype = PyList_GetItem(pyDtypeValues.get(), i);
        checkPyObjectIsNull(pyColDtype,"F-UDF-CL-SL-PYTHON-1128");
        PyPtr pyColDtypeString(PyObject_Str(pyColDtype));
        PyList_SET_ITEM(pyColumnDtypes.get(), i, pyColDtypeString.release());
    }

    return pyColumnDtypes.release();
}

void getOutputColumnTypes(PyObject *colTypes, std::vector<ColumnInfo>& colInfo)
{
    if (!PyList_Check(colTypes)) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-PYTHON-1077: getOutputColumnTypes(): colTypes is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    Py_ssize_t pyNumCols = PyList_Size(colTypes);
    for (Py_ssize_t i = 0; i < pyNumCols; i++) {
        PyObject *pyColType = PyList_GetItem(colTypes, i);
        checkPyObjectIsNull(pyColType,"F-UDF-CL-SL-PYTHON-1129");
        int colType = PyLong_AsLong(pyColType);
        if (colType < 0 && PyErr_Occurred())
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1078: getColumnInfo(): PyLong_AsLong error");

        colInfo.push_back(ColumnInfo(std::to_string(i), colType));
    }
}


static PyObject *getDataframe(PyObject *self, PyObject *args)
{
    PyObject *ctxIter = NULL;
    long numRows = 0;
    long startCol = 0;

    if (!PyArg_ParseTuple(args, "Oll", &ctxIter, &numRows, &startCol))
        return NULL;

    PyPtr pyDataFrame;
    try {
        PyPtr tableIter(PyObject_GetAttrString(ctxIter, "_exaiter__inp"));
        PyPtr colNames(PyObject_GetAttrString(ctxIter, "_exaiter__incolnames"));
        PyPtr pyInputType(PyObject_GetAttrString(ctxIter, "_exaiter__intype"));
        int inputType = PyLong_AsLong(pyInputType.get());
        if (inputType < 0 && PyErr_Occurred())
            throw std::runtime_error("F-UDF-CL-SL-PYTHON-1079: getDataframe(): PyLong_AsLong error");

        // Get script input type
        bool isSetInput = (static_cast<SWIGVMContainers::SWIGVM_itertype_e>(inputType) == SWIGVMContainers::MULTIPLE);
        // Get input column info
        std::vector<ColumnInfo> colInfo;
        getColumnInfo(ctxIter, colNames.get(), startCol, colInfo);
        // Get input data
        if (!isSetInput && numRows > 1)
            numRows = 1;
        bool isFinished = false;
        PyPtr pyData(getColumnData(colInfo, tableIter.get(), numRows, startCol, isSetInput, isFinished));
        if (isFinished) {
            int ok = PyObject_SetAttrString(ctxIter, "_exaiter__finished", Py_True);
            if (ok < 0)
                throw std::runtime_error("F-UDF-CL-SL-PYTHON-1080: getDataframe(): error setting exaiter.__finished");
        }
        // Create DataFrame
        pyDataFrame.reset(createDataFrame(pyData.get(), colInfo));
    }
    catch (std::exception &ex) {
        if (ex.what() && strlen(ex.what()))
            PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }

    return pyDataFrame.release();
}

static PyObject *emitDataframe(PyObject *self, PyObject *args)
{
    PyObject *ctxIter = NULL;
    PyObject *dataframe = NULL;

    if (!PyArg_ParseTuple(args, "OO", &ctxIter, &dataframe))
        return NULL;

    try {
        PyPtr resultHandler(PyObject_GetAttrString(ctxIter, "_exaiter__out"));
        PyPtr colTypes(PyObject_GetAttrString(ctxIter, "_exaiter__outcoltypes"));
        // Get output column info
        std::vector<ColumnInfo> colInfo;
        getOutputColumnTypes(colTypes.get(), colInfo);
        // Get NumPy types
        PyPtr pyNumpyTypes(getNumpyTypes(dataframe));
        // Emit output data
        emit(resultHandler.get(), colInfo, dataframe, pyNumpyTypes.get());
    }
    catch (std::exception &ex) {
        if (ex.what() && strlen(ex.what()))
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
