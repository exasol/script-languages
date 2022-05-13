#ifndef SWIG_COMMON_H
#define SWIG_COMMON_H


#include <vector>
#include <string>

namespace SWIGVMContainers {

enum VMTYPE {
    VM_UNSUPPORTED = 0,
    VM_PYTHON = 1,
    VM_SCHEME = 2,
    VM_JAVASCRIPT = 3,
    VM_R = 4,
    VM_EXTERNAL = 5,
    VM_JAVA = 6,
    VM_PLUGIN_LANGUAGE = 7,
    VM_BENCHMARK = 8
};


struct SWIGVMExceptionHandler
{
    SWIGVMExceptionHandler() { exthrowed = false; }
    ~SWIGVMExceptionHandler() { }
    void setException(const char* msg) {
        exmsg = msg; exthrowed = true;
    }
    void setException(const std::string& msg) {
        exmsg = msg; exthrowed = true;
    }
    std::string exmsg;
    bool exthrowed;
};


enum SWIGVM_datatype_e {
    UNSUPPORTED = 0,
    DOUBLE = 1,
    INT32 = 2,
    INT64 = 3,
    NUMERIC = 4,
    TIMESTAMP = 5,
    DATE = 6,
    STRING = 7,
    BOOLEAN = 8,
    INTERVALYM = 9,
    INTERVALDS = 10,
    GEOMETRY = 11
};


enum SWIGVM_itertype_e {
    EXACTLY_ONCE = 1,
    MULTIPLE = 2
};

enum single_call_function_id_e {
    SC_FN_NIL = 0,
    SC_FN_DEFAULT_OUTPUT_COLUMNS = 1,
    SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL = 2,
    SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC = 3,
    SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC = 4,
};

struct SWIGVM_columntype_t {
    SWIGVM_datatype_e type;
    std::string type_name;
    unsigned int len;
    unsigned int prec;
    unsigned int scale;
    SWIGVM_columntype_t(const SWIGVM_datatype_e t, const char *n, const unsigned int l, const unsigned int p, const unsigned int s):
        type(t), type_name(n), len(l), prec(p), scale(s) { }
    SWIGVM_columntype_t(const SWIGVM_datatype_e t, const char *n, const unsigned int l): type(t), type_name(n), len(l), prec(0), scale(0) { }
    SWIGVM_columntype_t(const SWIGVM_datatype_e t, const char *n): type(t), type_name(n), len(0), prec(0), scale(0) { }
    SWIGVM_columntype_t(): type(UNSUPPORTED), type_name("UNSUPPORTED"), len(0), prec(0), scale(0) { }
};

} //namespace SWIGVMContainers

#endif //SWIG_COMMON_H