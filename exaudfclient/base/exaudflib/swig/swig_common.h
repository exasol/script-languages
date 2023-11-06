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
    GEOMETRY = 11,
    HASHTYPE = 12
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


struct SWIGVM_params_t {
    uint64_t connection_id;
    SWIGVMExceptionHandler *exch;
    char *dbname;
    char *dbversion;
    char *script_name;
    char *script_schema;
    char *current_user;
    char *current_schema;
    char *scope_user;
    char *script_code;
    unsigned long long session_id;
    unsigned long statement_id;
    unsigned int node_count;
    unsigned int node_id;
    unsigned long long vm_id;
    VMTYPE vm_type;
    unsigned long long maximal_memory_limit;
    std::vector<std::string> *inp_names;
    std::vector<SWIGVM_columntype_t> *inp_types;
    SWIGVM_itertype_e inp_iter_type;
    bool inp_force_finish;
    std::vector<std::string> *out_names;
    std::vector<SWIGVM_columntype_t> *out_types;
    SWIGVM_itertype_e out_iter_type;
    bool m_allocate_params;
    std::vector<bool> *is_emitted;
    bool singleCallMode;
    std::string pluginName;
    std::string pluginURI;
    std::string outputAddress;
    SWIGVM_params_t():
        connection_id(0), exch(NULL), dbname(NULL), dbversion(NULL), script_name(NULL), script_schema(NULL), current_user(NULL),
        current_schema(NULL), scope_user(NULL), script_code(NULL),
        session_id(0), statement_id(0), node_count(0), node_id(0), vm_id(0),
        vm_type(VM_UNSUPPORTED), maximal_memory_limit(0),
        inp_names(NULL), inp_types(NULL), inp_iter_type(MULTIPLE), inp_force_finish(false),
        out_names(NULL), out_types(NULL), out_iter_type(MULTIPLE),
        m_allocate_params(false),
        is_emitted(NULL), singleCallMode(false), pluginName(""), pluginURI(""), outputAddress("")
    { }
    SWIGVM_params_t(const bool allocate_params):
        connection_id(0), exch(NULL), dbname(NULL), dbversion(NULL), script_name(NULL), script_schema(NULL), current_user(NULL),
        current_schema(NULL), scope_user(NULL), script_code(NULL),
        session_id(0), statement_id(0), node_count(0), node_id(0), vm_id(0),
        vm_type(VM_UNSUPPORTED), maximal_memory_limit(0),
        inp_names(allocate_params ? new std::vector<std::string>() : NULL),
        inp_types(allocate_params ? new std::vector<SWIGVM_columntype_t>() : NULL),
        inp_iter_type(MULTIPLE),
        inp_force_finish(false),
        out_names(allocate_params ? new std::vector<std::string>() : NULL),
        out_types(allocate_params ? new std::vector<SWIGVM_columntype_t>() : NULL),
        out_iter_type(MULTIPLE),
        m_allocate_params(allocate_params),
        is_emitted(allocate_params ? new std::vector<bool>() : NULL),
        singleCallMode(false), pluginName(""), pluginURI(""), outputAddress("")
    { }
    SWIGVM_params_t(const SWIGVM_params_t &p):
        connection_id(p.connection_id),
        dbname(p.dbname),
        dbversion(p.dbversion),
        script_name(p.script_name),
        script_schema(p.script_schema),
        current_user(p.current_user),
        current_schema(p.current_schema),
        scope_user(p.scope_user),
        script_code(p.script_code),
        session_id(p.session_id),
        statement_id(p.statement_id),
        node_count(p.node_count),
        node_id(p.node_id),
        vm_id(p.vm_id),
        vm_type(p.vm_type),
        maximal_memory_limit(p.maximal_memory_limit),
        inp_names(p.inp_names),
        inp_types(p.inp_types),
        inp_iter_type(p.inp_iter_type),
        inp_force_finish(p.inp_force_finish),
        out_names(p.out_names),
        out_types(p.out_types),
        out_iter_type(p.out_iter_type),
        m_allocate_params(false),
        is_emitted(p.is_emitted),
        pluginName(p.pluginName),
        pluginURI(p.pluginURI),
        outputAddress(p.outputAddress)
    {
        if (p.m_allocate_params)
            abort();
    }
    ~SWIGVM_params_t() {
        if (m_allocate_params) {
            if (inp_names != NULL) { delete inp_names; inp_names = NULL; }
            if (inp_types != NULL) { delete inp_types; inp_types = NULL; }
            if (out_names != NULL) { delete out_names; out_names = NULL; }
            if (out_types != NULL) { delete out_types; out_types = NULL; }
            if (is_emitted != NULL) { delete is_emitted; is_emitted = NULL; }
        }
    }
};

extern __thread SWIGVM_params_t *SWIGVM_params;

} //namespace SWIGVMContainers

#endif //SWIG_COMMON_H
