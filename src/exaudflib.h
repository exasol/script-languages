#ifndef EXAUDFLIB_H
#define EXAUDFLIB_H

#include <zmq.hpp>
#include <string>
#include <sstream>
#include "script_data_transfer_objects_wrapper.h"
#include <functional>
#include <map>
#include <mutex>
#include <iostream>

using namespace std;


void init_socket_name(const string socket_name);
bool send_init(zmq::socket_t &socket, const string client_name);
void send_close(zmq::socket_t &socket, const string &exmsg);
bool send_run(zmq::socket_t &socket);
bool send_return(zmq::socket_t &socket, std::string& result);
void send_undefined_call(zmq::socket_t &socket, const std::string& fn);
bool send_done(zmq::socket_t &socket);
void send_finished(zmq::socket_t &socket);

extern void* handle;
void* load_dynamic(const char* name);


namespace SWIGVMContainers {

//void socket_send(zmq::socket_t &socket, zmq::message_t &zmsg);
//bool socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error = false);

#define SWIG_MAX_VAR_DATASIZE 4000000

class SWIGVM;

enum VMTYPE {
    VM_UNSUPPORTED = 0,
    VM_PYTHON = 1,
    VM_SCHEME = 2,
    VM_JAVASCRIPT = 3,
    VM_R = 4,
    VM_EXTERNAL = 5,
    VM_JAVA = 6,
    VM_PLUGIN_LANGUAGE = 7
};


struct SWIGVMExceptionHandler
{
    SWIGVMExceptionHandler() { exthrowed = false; }
    ~SWIGVMExceptionHandler() { }
    void setException(const char* msg) {
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


struct SWIGVM_params_t {
    uint64_t connection_id;
    zmq::socket_t *sock;
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
        connection_id(0), sock(NULL),
        exch(NULL), dbname(NULL), dbversion(NULL), script_name(NULL), script_schema(NULL), current_user(NULL), current_schema(NULL), scope_user(NULL), script_code(NULL),
        session_id(0), statement_id(0), node_count(0), node_id(0), vm_id(0),
        vm_type(VM_UNSUPPORTED), maximal_memory_limit(0),
        inp_names(NULL), inp_types(NULL), inp_iter_type(MULTIPLE), inp_force_finish(false),
        out_names(NULL), out_types(NULL), out_iter_type(MULTIPLE),
        m_allocate_params(false),
        is_emitted(NULL), singleCallMode(false), pluginName(""), pluginURI(""), outputAddress("")
    { }
    SWIGVM_params_t(const bool allocate_params):
        connection_id(0), sock(NULL),
        exch(NULL), dbname(NULL), dbversion(NULL), script_name(NULL), script_schema(NULL), current_user(NULL), current_schema(NULL), scope_user(NULL), script_code(NULL),
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
        sock(p.sock),
        exch(p.exch),
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

class SWIGMetadata {
    SWIGMetadata* impl;
    typedef SWIGVMContainers::SWIGMetadata* (*CREATE_METADATA_FUN)();
    public:
        SWIGMetadata()
        {
            CREATE_METADATA_FUN create = (CREATE_METADATA_FUN)load_dynamic("create_SWIGMetaData");
            impl = create();
        }
        /* hack: use this constructor to avoid cycling loading of this class */
        SWIGMetadata(bool) {}

        virtual ~SWIGMetadata() { }
        virtual const char* databaseName() { return impl->databaseName(); }
        virtual const char* databaseVersion() { return impl->databaseVersion(); }
        virtual const char* scriptName() { return impl->scriptName(); }
        virtual const char* scriptSchema() { return impl->scriptSchema(); }
        virtual const char* currentUser() { return impl->currentUser(); }
        virtual const char* scopeUser() { return impl->scopeUser(); }
        virtual const char* currentSchema() {return impl->currentSchema();}
        virtual const char* scriptCode() { return impl->scriptCode(); }
        virtual const unsigned long long sessionID() { return impl->sessionID(); }
        virtual const char *sessionID_S() { return impl->sessionID_S(); }
        virtual const unsigned long statementID() { return impl->statementID(); }
        virtual const unsigned int nodeCount() { return impl->nodeCount(); }
        virtual const unsigned int nodeID() { return impl->nodeID(); }
        virtual const unsigned long long vmID() { return impl->vmID(); }
        virtual const unsigned long long memoryLimit() { return impl->memoryLimit(); }
        virtual const VMTYPE vmType() { return impl->vmType(); }
        virtual const char *vmID_S() { return impl->vmID_S(); }
        virtual const ExecutionGraph::ConnectionInformationWrapper connectionInformation(const char* connection_name){
            return impl->connectionInformation(connection_name);
        }
        virtual const char* moduleContent(const char* name) {return impl->moduleContent(name);}
        virtual const unsigned int inputColumnCount() { return impl->inputColumnCount(); }
        virtual const char *inputColumnName(unsigned int col) { return impl->inputColumnName(col);}
        virtual const SWIGVM_datatype_e inputColumnType(unsigned int col) { return impl->inputColumnType(col);}
        virtual const char *inputColumnTypeName(unsigned int col) { return impl->inputColumnTypeName(col); }
        virtual const unsigned int inputColumnSize(unsigned int col) {return impl->inputColumnSize(col);}
        virtual const unsigned int inputColumnPrecision(unsigned int col) { return impl->inputColumnPrecision(col); }
        virtual const unsigned int inputColumnScale(unsigned int col) { return impl->inputColumnScale(col);}
        virtual const SWIGVM_itertype_e inputType() { return impl->inputType();}
        virtual const unsigned int outputColumnCount() { return impl->outputColumnCount(); }
        virtual const char *outputColumnName(unsigned int col) { return impl->outputColumnName(col); }
        virtual const SWIGVM_datatype_e outputColumnType(unsigned int col) { return impl->outputColumnType(col); }
        virtual const char *outputColumnTypeName(unsigned int col) { return impl->outputColumnTypeName(col); }
        virtual const unsigned int outputColumnSize(unsigned int col) { return impl->outputColumnSize(col); }
        virtual const unsigned int outputColumnPrecision(unsigned int col) {return impl->outputColumnPrecision(col);}
        virtual const unsigned int outputColumnScale(unsigned int col) {return impl->outputColumnScale(col);}
        virtual const SWIGVM_itertype_e outputType() { return impl->outputType(); }
        virtual const bool isEmittedColumn(unsigned int col){ return impl->isEmittedColumn(col);}
        virtual const char* checkException() {return impl->checkException();}
        virtual const char* pluginLanguageName() {return impl->pluginLanguageName();}
        virtual const char* pluginURI() {return impl->pluginURI();}
        virtual const char* outputAddress() {return impl->outputAddress();}
};


class AbstractSWIGTableIterator {
public:
    virtual ~AbstractSWIGTableIterator() {}
    virtual void reinitialize()=0;
    virtual bool next()=0;
    virtual bool eot()=0;
    virtual void reset()=0;
    virtual unsigned long restBufferSize()=0;
    virtual unsigned long rowsCompleted()=0;
    virtual unsigned long rowsInGroup()=0;
    virtual double getDouble(unsigned int col)=0;
    virtual const char *getString(unsigned int col, size_t *length = NULL)=0;
    virtual int32_t getInt32(unsigned int col)=0;
    virtual int64_t getInt64(unsigned int col)=0;
    virtual const char *getNumeric(unsigned int col)=0;
    virtual const char *getTimestamp(unsigned int col)=0;
    virtual const char *getDate(unsigned int col)=0;
    virtual bool getBoolean(unsigned int col)=0;
    virtual bool wasNull()=0;
    virtual uint64_t get_current_row()=0;
    virtual const char* checkException()=0;
};



class SWIGTableIterator { //: public AbstractSWIGTableIterator {
    typedef SWIGVMContainers::AbstractSWIGTableIterator* (*CREATE_TABLEITERATOR_FUN)();

    AbstractSWIGTableIterator* impl=nullptr;
public:
    SWIGTableIterator()
    {
        CREATE_TABLEITERATOR_FUN creator = (CREATE_TABLEITERATOR_FUN)load_dynamic("create_SWIGTableIterator");
        impl = creator();
    }

    virtual ~SWIGTableIterator() {
        if (impl!=nullptr) {
            delete impl;
        }
    }
    virtual void reinitialize() { impl->reinitialize();}
    virtual bool next() { return impl->next(); }
    virtual bool eot() { return impl->eot(); }
    virtual void reset() { return impl->reset(); }
    virtual unsigned long restBufferSize() { return impl->restBufferSize();}
    virtual unsigned long rowsCompleted() { return impl->rowsCompleted();}
    virtual unsigned long rowsInGroup() { return impl->rowsInGroup();}
    virtual double getDouble(unsigned int col) { return impl->getDouble(col);}
    virtual const char *getString(unsigned int col, size_t *length = NULL) {return impl->getString(col, length);}
    virtual int32_t getInt32(unsigned int col) {return impl->getInt32(col);}
    virtual int64_t getInt64(unsigned int col) {return impl->getInt64(col);}
    virtual const char *getNumeric(unsigned int col) {return impl->getNumeric(col);}
    virtual const char *getTimestamp(unsigned int col) {return impl->getTimestamp(col);}
    virtual const char *getDate(unsigned int col) {return impl->getDate(col);}
    virtual bool getBoolean(unsigned int col) {return impl->getBoolean(col);}
    virtual bool wasNull() { return impl->wasNull();}
    virtual uint64_t get_current_row() {return impl->get_current_row();}
    const char* checkException() {return impl->checkException();}
};



class SWIGRAbstractResultHandler {
public:
    virtual ~SWIGRAbstractResultHandler() {};
    virtual void reinitialize()=0;
    virtual unsigned long rowsEmited()=0;
    virtual void flush()=0;
    virtual bool next()=0;
    virtual void setDouble(unsigned int col, const double v)=0;
    virtual void setString(unsigned int col, const char *v, size_t l)=0;
    virtual void setInt32(unsigned int col, const int32_t v)=0;
    virtual void setInt64(unsigned int col, const int64_t v)=0;
    virtual void setNumeric(unsigned int col, const char *v)=0;
    virtual void setTimestamp(unsigned int col, const char *v)=0;
    virtual void setDate(unsigned int col, const char *v)=0;
    virtual void setBoolean(unsigned int col, const bool v)=0;
    virtual void setNull(unsigned int col)=0;
    virtual const char* checkException()=0;
};


class SWIGResultHandler { //: public SWIGRAbstractResultHandler {
    SWIGRAbstractResultHandler* impl=nullptr;
    typedef SWIGVMContainers::SWIGRAbstractResultHandler* (*CREATE_RESULTHANDLER_FUN)(SWIGVMContainers::SWIGTableIterator*);
public:
    SWIGResultHandler(SWIGTableIterator* table_iterator)
    {
        CREATE_RESULTHANDLER_FUN creator = (CREATE_RESULTHANDLER_FUN)load_dynamic("create_SWIGResultHandler");
        impl = creator(table_iterator);
    }

    virtual ~SWIGResultHandler() {
        if (impl!=nullptr) {
            delete impl;
        }
    }
    virtual void reinitialize() {impl->reinitialize(); }
    virtual unsigned long rowsEmited() {return impl->rowsEmited();}
    virtual void flush() {impl->flush();}
    virtual bool next() {return impl->next();}
    virtual void setDouble(unsigned int col, const double v) {impl->setDouble(col, v);}
    virtual void setString(unsigned int col, const char *v, size_t l) {impl->setString(col,v,l);}
    virtual void setInt32(unsigned int col, const int32_t v) {impl->setInt32(col,v);}
    virtual void setInt64(unsigned int col, const int64_t v) {impl->setInt64(col,v);}
    virtual void setNumeric(unsigned int col, const char *v) {impl->setNumeric(col,v);}
    virtual void setTimestamp(unsigned int col, const char *v) {impl->setTimestamp(col,v);}
    virtual void setDate(unsigned int col, const char *v) {impl->setDate(col,v);}
    virtual void setBoolean(unsigned int col, const bool v) {impl->setBoolean(col,v);}
    virtual void setNull(unsigned int col) {impl->setNull(col);}
    const char* checkException() {return impl->checkException();}
};




class SWIGVM {
    public:
        struct exception : public std::exception {
            exception(const char *reason): m_reason(reason) { }
            virtual ~exception() throw() { }
            const char* what() const throw() { return m_reason.c_str(); }
            private:
                std::string m_reason;
        };
        SWIGVM() { }
        virtual ~SWIGVM() { }
        virtual void shutdown() {};
        virtual void destroy() {};
        virtual bool run() = 0;
        bool run_() {
            try {
                return run();
            } catch (SWIGVM::exception& ex) {
                std::cerr << "SWGVM run_: caught: " << ex.what();
                std::lock_guard<mutex> lck(exception_msg_mtx);
                exception_msg = ex.what();
                return true; /* we are done */
            }
        }
        virtual bool useZmqSocketLocks() {return false;}
        virtual std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) = 0;
        string exception_msg;
        mutex exception_msg_mtx;
        string calledUndefinedSingleCall;

};
//struct swig_undefined_single_call_exception: public std::exception
//{
//    swig_undefined_single_call_exception(const std::string& fn): m_fn(fn) { }
//    virtual ~swig_undefined_single_call_exception() throw() { }
//    const std::string fn() const {return m_fn;}
//    const char* what() const throw() {
//        std::stringstream sb;
//        sb << "Undefined in UDF: " << m_fn;
//        return sb.str().c_str();
//    }
// private:
//    const std::string m_fn;
//};
#ifdef ENABLE_PYTHON_VM
class PythonVMImpl;

class PythonVM: public SWIGVM {
    public:
//        struct exception: SWIGVM::exception {
//            exception(const char *reason): SWIGVM::exception(reason) { }
//            virtual ~exception() throw() { }
//        };
        PythonVM(bool checkOnly);
        virtual ~PythonVM() {};
        virtual void shutdown();
        virtual bool run();        
        virtual std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
    private:
        PythonVMImpl *m_impl;
};
#endif


#ifdef ENABLE_R_VM
class RVMImpl;

class RVM: public SWIGVM {
    public:
//        struct exception: SWIGVM::exception {
//            exception(const char *reason): SWIGVM::exception(reason) { }
//            virtual ~exception() throw() { }
//        };
        RVM(bool checkOnly);
        virtual ~RVM() {};
        virtual bool run();
        virtual void shutdown();
        virtual std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
    private:
        RVMImpl *m_impl;
};

#endif

#ifdef ENABLE_JAVA_VM
class JavaVMImpl;

class JavaVMach: public SWIGVM {
    public:
//        struct exception: SWIGVM::exception {
//            exception(const char *reason): SWIGVM::exception(reason) { }
//            virtual ~exception() throw() { }
//        };
        JavaVMach(bool checkOnly);
        virtual ~JavaVMach() {}
        virtual void shutdown();
        virtual bool run();
        virtual std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
    private:
        JavaVMImpl *m_impl;
};

#endif


#ifdef ENABLE_STREAMING_VM
class StreamingVM: public SWIGVM {
    public:
        struct exception: SWIGVM::exception {
            exception(const char *reason): SWIGVM::exception(reason) { }
            virtual ~exception() throw() { }
        };
        StreamingVM(bool checkOnly);
        virtual ~StreamingVM() {};
        virtual void shutdown();
        virtual bool run();
        virtual std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
        virtual bool useZmqSocketLocks() {return true;}
    private:
        SWIGMetadata meta;
        SWIGTableIterator inp;
        SWIGResultHandler outp;
        void importScripts();
        map<SWIGVM_datatype_e, std::function<void(ostream&str, unsigned int col)> > csvPrinters;
        map<SWIGVM_datatype_e, std::function<void(string& str, unsigned int col)> > csvReaders;
        void inputToCSV(ostream&str);
        bool CSVToOutput(istream&str);
        string readBuffer;
        // returns true if a an entry could be read
        bool readCSVValue(istream&str, unsigned int column);
};
#endif



} // namespace swigvm container


#endif // EXAUDFLIB_H
