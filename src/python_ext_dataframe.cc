#include "exaudflib.h"

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

#if 0
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
#endif

#define PY_INT (NPY_USERDEF+1)
#define PY_STR (NPY_USERDEF+2)

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
    {"py_int", PY_INT},
    {"py_str", PY_STR},
    {"object", NPY_OBJECT}
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
    ColumnInfo(std::string const& name, long type) {
        this->name = name;
        this->type = static_cast<SWIGVMContainers::SWIGVM_datatype_e>(type);
    }

    std::string name;
    SWIGVMContainers::SWIGVM_datatype_e type;
};



void getColumnInfo(PyObject *ctxIter, PyObject *exaMeta, bool isInput, std::vector<ColumnInfo>& colInfo)
{
    const char *ctxColumnTypeList = isInput ? "_exaiter__incoltypes" : "_exaiter__outcoltypes";
    const char *metaColumnList = isInput ? "input_columns" : "output_columns";

    PyPtr pyColTypes(PyObject_GetAttrString(ctxIter, ctxColumnTypeList));
    checkPyPtrIsNull(pyColTypes);
    if (!PyList_Check(pyColTypes.get())) {
        std::stringstream ss;
        ss << "getColumnInfo: " << ctxColumnTypeList << " is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    PyPtr pyMetaCols(PyObject_GetAttrString(exaMeta, metaColumnList));
    checkPyPtrIsNull(pyMetaCols);
    if (!PyList_Check(pyMetaCols.get())) {
        std::stringstream ss;
        ss << "getColumnInfo: " << metaColumnList << " is not a list";
        throw std::runtime_error(ss.str().c_str());
    }

    Py_ssize_t pyNumCols = PyList_Size(pyColTypes.get());
    if (pyNumCols != PyList_Size(pyMetaCols.get())) {
        std::stringstream ss;
        ss << "getColumnInfo: ";
        ss << ctxColumnTypeList << " has length " << pyNumCols << ", but ";
        ss << metaColumnList << " has length " << PyList_Size(pyMetaCols.get());
        throw std::runtime_error(ss.str().c_str());
    }

    for (Py_ssize_t i = 0; i < pyNumCols; i++) {
        PyPtr pyColType(PyList_GetItem(pyColTypes.get(), i));
        checkPyPtrIsNull(pyColType);
        long colType = PyLong_AsLong(pyColType.get());
        if (colType < 0 && PyErr_Occurred())
            throw std::runtime_error("getColumnInfo(): PyLong_AsLong error");

        PyPtr pyMetaCol(PyList_GetItem(pyMetaCols.get(), i));
        checkPyPtrIsNull(pyMetaCol);
        PyPtr pyColName(PyObject_GetAttrString(pyMetaCol.get(), "name"));
        checkPyPtrIsNull(pyColName);
        const char *colName = PyUnicode_AsUTF8(pyColName.get());
        if (!colName)
            throw std::runtime_error("");

        colInfo.push_back(ColumnInfo(std::string(colName), colType));
    }
}

PyObject *getColumnData(std::vector<ColumnInfo>& colInfo, PyObject *tableIter, long numRows)
{
    const long numCols = colInfo.size();
    std::vector<std::pair<PyPtr, PyPtr>> pyColGetMethods;

    for (long i = 0; i < numCols; i++) {
        PyPtr pyColNum(PyLong_FromLong(i));
        checkPyPtrIsNull(pyColNum);

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
                methodName = "getNumeric";
                break;
            case SWIGVMContainers::STRING:
                methodName = "getString";
                break;
            case SWIGVMContainers::BOOLEAN:
                methodName = "getBoolean";
                break;
            case SWIGVMContainers::DATE:
                methodName = "getDate";
                break;
            case SWIGVMContainers::TIMESTAMP:
                methodName = "getTimestamp";
                break;
            default:
            {
                std::stringstream ss;
                ss << "getColumnData(): unexpected type " << colInfo[i].type;
                throw std::runtime_error(ss.str().c_str());
            }
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
            PyPtr pyVal(PyObject_CallMethodObjArgs(tableIter, pyColGetMethods[c].second.get(), pyColGetMethods[c].first.get(), NULL));
            if (!pyVal) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                std::stringstream ss;
                ss << "getColumnData(): Error fetching value for row " << r << ", column " << c << ": ";
                ss << PyUnicode_AsUTF8(pvalue);
                throw std::runtime_error(ss.str().c_str());
            }

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(tableIter, pyCheckExceptionMethodName.get(), NULL));
            checkPyPtrIsNull(pyCheckException);
            if (pyCheckException.get() != Py_None) {
                const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
                if (exMsg) {
                    std::stringstream ss;
                    ss << "getColumnData(): get row " << r << ", column " << c << exMsg;
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            PyPtr pyWasNull(PyObject_CallMethodObjArgs(tableIter, pyWasNullMethodName.get(), NULL));
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

        PyPtr pyNext(PyObject_CallMethodObjArgs(tableIter, pyNextMethodName.get(), NULL));
        checkPyPtrIsNull(pyNext);

        PyPtr pyCheckException(PyObject_CallMethodObjArgs(tableIter, pyCheckExceptionMethodName.get(), NULL));
        checkPyPtrIsNull(pyCheckException);
        if (pyCheckException.get() != Py_None) {
            const char *exMsg = PyUnicode_AsUTF8(pyCheckException.get());
            if (exMsg) {
                std::stringstream ss;
                ss << "getColumnData(): next(): " << exMsg;
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

void emit(PyObject *exaMeta, PyObject *resultHandler, std::vector<ColumnInfo>& colInfo, PyObject *dataframe, PyObject *numpyTypes)
{
    std::vector<std::pair<PyPtr, PyPtr>> pyColSetMethods;
    for (unsigned int i = 0; i < colInfo.size(); i++) {
        PyPtr pyColNum(PyLong_FromLong(i));
        checkPyPtrIsNull(pyColNum);

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
                ss << "emit(): unexpected type " << colInfo[i].type;
                throw std::runtime_error(ss.str().c_str());
            }
        }
        PyPtr pyMethodName(PyUnicode_FromString(methodName.c_str()));
        checkPyPtrIsNull(pyMethodName);

        pyColSetMethods.push_back(std::make_pair(std::move(pyColNum), std::move(pyMethodName)));
    }

    // Get data type info
    std::vector<std::pair<std::string, int>> colTypes;
    PyPtr numpyTypeIter(PyObject_GetIter(numpyTypes));
    checkPyPtrIsNull(numpyTypeIter);

    std::map<std::string, int>::iterator it;
    for (PyPtr numpyType(PyIter_Next(numpyTypeIter.get())); numpyType.get(); numpyType.reset(PyIter_Next(numpyTypeIter.get()))) {
        const char *typeName = PyUnicode_AsUTF8(numpyType.get());
        it = typeMap.find(typeName);
        if (it != typeMap.end()) {
            colTypes.push_back(*it);
        }
        else {
            std::stringstream ss;
            ss << "emit: unexpected type: " << typeName;
            throw std::runtime_error(ss.str().c_str());
        }
    }

    PyPtr data(PyObject_GetAttrString(dataframe, "values"));
    checkPyPtrIsNull(data);


    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(data.get(),
                                                            NPY_OBJECT,
                                                            NPY_ARRAY_IN_ARRAY));
    int numRows = PyArray_DIM(array, 0);
    int numCols = PyArray_DIM(array, 1);

    std::vector<PyPtr> columnArrays;
    // Transpose to column-major
    PyObject *colArray = PyArray_Transpose(array, NULL);
    for (int c = 0; c < numCols; c++) {
        PyPtr pyStart(PyLong_FromLong(c));
        PyPtr pyStop(PyLong_FromLong(c + 1));
        PyPtr none(Py_None);
        PyPtr slice(PySlice_New(pyStart.get(), pyStop.get(), none.get()));
        checkPyPtrIsNull(slice);
        PyPtr arr(PyObject_GetItem(colArray, slice.get()));
        checkPyPtrIsNull(arr);

        PyPtr pyZero(PyLong_FromLong(0L));
        arr.reset(PyObject_GetItem(arr.get(), pyZero.get()));
        checkPyPtrIsNull(arr);

        PyPtr asType (PyObject_GetAttrString(arr.get(), "astype"));
        checkPyPtrIsNull(asType);
        PyPtr keywordArgs(PyDict_New());
        checkPyPtrIsNull(keywordArgs);
        PyDict_SetItemString(keywordArgs.get(), "copy", Py_False);
        PyPtr funcArgs(Py_BuildValue("(s)", colTypes[c].first.c_str()));
        checkPyPtrIsNull(funcArgs);
        PyPtr scalarArr(PyObject_Call(asType.get(), funcArgs.get(), keywordArgs.get()));
        checkPyPtrIsNull(scalarArr);

        if (colTypes[c].second == NPY_OBJECT) {
            // Convert numpy array to python list
            PyPtr pyList(PyObject_CallMethod(scalarArr.get(), "tolist", NULL));
            checkPyPtrIsNull(pyList);
            if (!PyList_Check(pyList.get())) {
                std::stringstream ss;
                ss << "emit(): column array " << c << " is not a list";
                throw std::runtime_error(ss.str().c_str());
            }

            // Get type of first item in list
            PyPtr pyVal(PyList_GetItem(pyList.get(), 0));
            checkPyPtrIsNull(pyVal);
            std::string pyTypeName("py_");
            pyTypeName.append(Py_TYPE(pyVal.get())->tp_name);

            // Update type in column type info
            std::map<std::string, int>::iterator userDefIt;
            userDefIt = typeMap.find(pyTypeName);
            if (userDefIt != typeMap.end()) {
                colTypes[c] = *userDefIt;
            }
            else {
                std::stringstream ss;
                ss << "emit: unexpected python type: " << pyTypeName;
                throw std::runtime_error(ss.str().c_str());
            }

            columnArrays.push_back(std::move(pyList));
        }
        else {
            columnArrays.push_back(std::move(scalarArr));
        }
    }

    PyPtr pyNextMethodName(PyUnicode_FromString("next"));
    checkPyPtrIsNull(pyNextMethodName);
    PyPtr pyCheckExceptionMethodName(PyUnicode_FromString("checkException"));
    checkPyPtrIsNull(pyCheckExceptionMethodName);

    PyPtr arrayIter(PyArray_IterNew(data.get()));
    checkPyPtrIsNull(arrayIter);
    PyArrayIterObject *iter = reinterpret_cast<PyArrayIterObject*>(arrayIter.get());

    PyPtr pyValue;
    PyPtr pyResult;
    for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
            switch (colTypes[c].second) {
                case NPY_INT64:
                case NPY_UINT64:
                {
                    int64_t value = *((int64_t*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
                    //ss << "INT64: " << value << "; ";
                    //break;

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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    //std::stringstream ss;
                    //ss << "XXXX types: '" << Py_TYPE(pyColSetMethods[c].first.get())->tp_name << "', '" << Py_TYPE(pyValue.get())->tp_name << "' XXXXX";
                    //throw std::runtime_error(ss.str().c_str());
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_INT32:
                case NPY_UINT32:
                {
                    int32_t value = *((int32_t*)(iter->dataptr));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_INT16:
                case NPY_UINT16:
                {
                    int16_t value = *((int16_t*)(iter->dataptr));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_INT8:
                case NPY_UINT8:
                {
                    int8_t value = *((int8_t*)(iter->dataptr));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_FLOAT64:
                {
                    double value = *((double*)(iter->dataptr));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_FLOAT32:
                {
                    double value = *((float*)(iter->dataptr));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_FLOAT16:
                {
                    double value = static_cast<double>(*((uint16_t*)(iter->dataptr)));
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
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case NPY_BOOL:
                {
                    bool value = *((bool*)PyArray_GETPTR1((PyArrayObject*)(columnArrays[c].get()), r));
                    //ss << "BOOL: " << value << "; ";
                    //break;

                    switch (colInfo[c].type) {
                        case SWIGVMContainers::BOOLEAN:
                            pyValue.reset(value ? Py_True : Py_False);
                            break;
                        default:
                        {
                            std::stringstream ss;
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    checkPyPtrIsNull(pyValue);
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case PY_INT:
                {
                    PyPtr pyInt(PyList_GetItem(columnArrays[c].get(), r));
                    checkPyPtrIsNull(pyInt);

                    switch (colInfo[c].type) {
                        case SWIGVMContainers::INT64:
                        case SWIGVMContainers::INT32:
                            pyValue.reset(pyInt.get());
                            break;
                        case SWIGVMContainers::NUMERIC:
                            pyValue.reset(PyObject_Str(pyInt.get()));
                            break;
                        case SWIGVMContainers::DOUBLE:
                        {
                            double value = PyFloat_AsDouble(pyInt.get());
                            if (value < 0 && PyErr_Occurred())
                                throw std::runtime_error("emit() PY_INT: PyFloat_AsDouble error");
                            pyValue.reset(PyFloat_FromDouble(value));
                            break;
                        }
                        default:
                        {
                            std::stringstream ss;
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                    break;
                }
                case PY_STR:
                {
                    PyPtr pyValue(PyList_GetItem(columnArrays[c].get(), r));
                    checkPyPtrIsNull(pyValue);

                    switch (colInfo[c].type) {
                        case SWIGVMContainers::NUMERIC:
                            pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), NULL));
                            break;
                        case SWIGVMContainers::STRING: {
                            Py_ssize_t size = -1;
                            const char *str = PyUnicode_AsUTF8AndSize(pyValue.get(), &size);
                            if (!str && size < 0)
                                throw std::runtime_error("");
                            PyPtr pySize(PyLong_FromSsize_t(size));
                            checkPyPtrIsNull(pySize);
                            pyResult.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyValue.get(), pySize.get(), NULL));
                            break;
                        }
                        default:
                        {
                            std::stringstream ss;
                            ss << "emit column " << c << " of type " << colInfo[c].type << " but data given have type " << colTypes[c].first;
                            throw std::runtime_error(ss.str().c_str());
                        }
                    }
                    break;
                }
                case NPY_OBJECT:
                {
                    std::stringstream ss;
                    ss << "emit column " << c << ": OBJECT not supported";
                    throw std::runtime_error(ss.str().c_str());

                    // Have to check for DATE, NUMERIC, etc.
                    //const char *val = (char*)(iter->dataptr);
                    //PyPtr pyVal(PyUnicode_FromString(val));
#if 0
                    checkPyPtrIsNull(pyVal);
                    pyVal.reset(PyObject_CallMethodObjArgs(resultHandler, pyColSetMethods[c].second.get(), pyColSetMethods[c].first.get(), pyVal.get(), strlen(val), NULL));
#endif
                    break;
                }
                default:
                {
                    std::stringstream ss;
                    ss << "emit: unexpected type: " << colTypes[c].first;
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            if (!pyResult) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                if (pvalue) {
                    std::stringstream ss;
                    ss << "emit(): Error setting value for row " << r << ", column " << c << ": ";
                    ss << PyUnicode_AsUTF8(pvalue);
                    throw std::runtime_error(ss.str().c_str());
                }
            }

            PyPtr pyCheckException(PyObject_CallMethodObjArgs(resultHandler, pyCheckExceptionMethodName.get(), NULL));
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

        PyPtr pyNext(PyObject_CallMethodObjArgs(resultHandler, pyNextMethodName.get(), NULL));
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
        PyPtr tableIter(PyObject_GetAttrString(ctxIter, "_exaiter__inp"));
        checkPyPtrIsNull(tableIter);
        // Get input column info
        std::vector<ColumnInfo> colInfo;
        getColumnInfo(ctxIter, exaMeta, true, colInfo);
        // Get input data
        pyData.reset(getColumnData(colInfo, tableIter.get(), numOutRows));
    }
    catch (std::exception &ex) {
        if (ex.what() && strlen(ex.what()))
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
        PyPtr resultHandler(PyObject_GetAttrString(ctxIter, "_exaiter__out"));
        checkPyPtrIsNull(resultHandler);
        // Get output column info
        std::vector<ColumnInfo> colInfo;
        getColumnInfo(ctxIter, exaMeta, false, colInfo);
        // Emit output data
        emit(exaMeta, resultHandler.get(), colInfo, dataframe, numpyTypes);
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
