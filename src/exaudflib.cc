#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#define DONT_EXPOSE_SWIGVM_PARAMS
#include "exaudflib.h"
#undef DONT_EXPOSE_SWIGVM_PARAMS
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include <fcntl.h>
#include <fstream>
#include "script_data_transfer_objects.h"
#include <functional>

// swig lib
#include <limits>
#include "zmqcontainer.pb.h"
#include "script_data_transfer_objects_wrapper.h"
#include <unistd.h>

#include <mutex>

#ifdef PROTEGRITY_PLUGIN_CLIENT
#include <protegrityclient.h>
#endif

using namespace SWIGVMContainers;
using namespace std;
using namespace google::protobuf;

#ifndef PROTEGRITY_PLUGIN_CLIENT
__thread SWIGVM_params_t* SWIGVMContainers::SWIGVM_params; // this is not used in the file, but defined to satisfy the "extern" requirement from exaudflib.h
#endif

static SWIGVM_params_t * SWIGVM_params_ref = nullptr;


static pid_t my_pid; //parent_pid,

static const char *socket_name_str;

static string output_buffer;
static SWIGVMExceptionHandler exchandler;

//static exascript_vmtype vm_type;
static exascript_request request;
static exascript_response response;

static string g_database_name;
static string g_database_version;
static string g_script_name;
static string g_script_schema;
static string g_current_user;
static string g_scope_user;
static string g_current_schema;
static string g_source_code;
static unsigned long long g_session_id;
static unsigned long g_statement_id;
static unsigned int g_node_count;
static unsigned int g_node_id;
static unsigned long long g_vm_id;
static bool g_singleCallMode;
static single_call_function_id_e g_singleCallFunction;
static ExecutionGraph::ImportSpecification g_singleCall_ImportSpecificationArg;
static ExecutionGraph::ExportSpecification g_singleCall_ExportSpecificationArg;
static ExecutionGraph::StringDTO g_singleCall_StringArg;
static bool remote_client;

#ifndef NDEBUG
#define SWIGVM_LOG_CLIENT
#endif
//#define SWIGVM_LOG_CLIENT
//#define LOG_COMMUNICATION

pthread_t check_thread;

extern "C" {
void set_SWIGVM_params(SWIGVM_params_t* p) {
    SWIGVM_params_ref = p;
}
}

void init_socket_name(const char* the_socket_name) {
    socket_name_str = the_socket_name;
}

static void external_process_check()
{
    if (remote_client) return;
    if (::access(&(socket_name_str[6]), F_OK) != 0) {
        ::sleep(1); // give me a chance to die with my parent process
        cerr << "exaudfclient aborting ... cannot access socket file " << socket_name_str+6 << "." << endl;
#ifdef SWIGVM_LOG_CLIENT
        cerr << "### SWIGVM aborting with name '" << socket_name_str
             << "' (" << ::getppid() << ',' << ::getpid() << ')' << endl;
#endif
        ::abort();
    }
}



void set_remote_client(bool value) {
    remote_client = value;
}

bool get_remote_client() {
    return remote_client;
}


static bool keep_checking = true;

void *check_thread_routine(void* data)
{
    while(keep_checking) {
        external_process_check();
        ::usleep(100000);
    }
    return NULL;

}

void start_check_thread() {
    if (!remote_client)
        pthread_create(&check_thread, NULL, check_thread_routine, NULL);
}

void stop_check_thread() {
    keep_checking = false;
}

void cancel_check_thread() {
    ::pthread_cancel(check_thread);
}

mutex zmq_socket_mutex;
static bool use_zmq_socket_locks = false;


void socket_send(zmq::socket_t &socket, zmq::message_t &zmsg)
{
#ifdef LOG_COMMUNICATION
    stringstream sb;
    uint32_t len = zmsg.size();
    sb << "/tmp/zmqcomm_log_" << ::getpid() << "_send.data";
    int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
    if (fd >= 0) {
        if (::write(fd, &len, sizeof(uint32_t)) == -1 ) {perror("Log communication");}
        if (::write(fd, zmsg.data(), len) == -1) {perror("Log communication");}
        ::close(fd);
    }
#endif
    for (;;) {
        try {
            if (use_zmq_socket_locks) {
                zmq_socket_mutex.lock();
            }
            if (socket.send(zmsg) == true) {
                if (use_zmq_socket_locks) {
                    zmq_socket_mutex.unlock();
                }
                return;
            }
            external_process_check();
        } catch (std::exception &err) {
            external_process_check();
        } catch (...) {
            external_process_check();
        }
        if (use_zmq_socket_locks) {
            zmq_socket_mutex.unlock();
        }
        ::usleep(100000);
    }
    if (use_zmq_socket_locks) {
        zmq_socket_mutex.unlock();
    }
}

bool socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error=false)
{
    for (;;) {
        try {
            if (use_zmq_socket_locks) {
            zmq_socket_mutex.lock();
            }
            if (socket.recv(&zmsg) == true) {
#ifdef LOG_COMMUNICATION
                stringstream sb;
                uint32_t len = zmsg.size();
                sb << "/tmp/zmqcomm_log_" << ::getpid() << "_recv.data";
                int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
                if (fd >= 0) {
                    if (::write(fd, &len, sizeof(uint32_t)) == -1) {perror("Log communication");}
                    if (::write(fd, zmsg.data(), len) == -1) {perror("Log communication");}
                    ::close(fd);
                }
#endif
                if (use_zmq_socket_locks) {
                    zmq_socket_mutex.unlock();
                }
                return true;
            }
            external_process_check();
        } catch (std::exception &err) {
            external_process_check();

        } catch (...) {
            external_process_check();
        }
        if (use_zmq_socket_locks) {
            zmq_socket_mutex.unlock();
        }
        if (return_on_error) return false;
        ::usleep(100000);
    }
    if (use_zmq_socket_locks) {
        zmq_socket_mutex.unlock();
    }
    return false;
}

bool send_init(zmq::socket_t &socket, const string client_name)
{
    request.Clear();
    request.set_type(MT_CLIENT);
    request.set_connection_id(0);
    exascript_client *req = request.mutable_client();
    req->set_client_name(client_name);
    if (!request.SerializeToString(&output_buffer)) {
        exchandler.setException("Communication error: failed to serialize data");
        return false;
    }
    zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
    socket_send(socket, zmsg);

    zmq::message_t zmsgrecv;
    response.Clear();
    if (!socket_recv(socket, zmsgrecv, true))
        return false;
    if (!response.ParseFromArray(zmsgrecv.data(), zmsgrecv.size())) {
        exchandler.setException("Failed to parse data");
        return false;
    }

    SWIGVM_params_ref->connection_id = response.connection_id();
#ifdef SWIGVM_LOG_CLIENT
    stringstream sb; sb << std::hex << SWIGVM_params_ref->connection_id;
    cerr << "### SWIGVM connected with id " << sb.str() << endl;
#endif
    if (response.type() == MT_CLOSE) {
        if (response.close().has_exception_message())
            exchandler.setException(response.close().exception_message().c_str());
        else exchandler.setException("Connection closed by server");
        return false;
    }
    if (response.type() != MT_INFO) {
        exchandler.setException("Wrong message type, should be MT_INFO");
        return false;
    }
    const exascript_info &rep = response.info();
    g_database_name = rep.database_name();
    g_database_version = rep.database_version();
    g_script_name = rep.script_name();
    g_script_schema = rep.script_schema();
    g_current_user = rep.current_user();
    g_scope_user = rep.scope_user();
    if (g_scope_user.size()==0) {         // for backward compatibility when testing with EXASOL 6.0.8 installations at OTTO Brain
        g_scope_user=g_current_user;
    }
    g_current_schema = rep.current_schema();
    g_source_code = rep.source_code();
    g_session_id = rep.session_id();
    g_statement_id = rep.statement_id();
    g_node_count = rep.node_count();
    g_node_id = rep.node_id();
    g_vm_id = rep.vm_id();
    //vm_type = rep.vm_type();


    SWIGVM_params_ref->maximal_memory_limit = rep.maximal_memory_limit();
    struct rlimit d;
    d.rlim_cur = d.rlim_max = rep.maximal_memory_limit();
    if (setrlimit(RLIMIT_RSS, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        cerr << "WARNING: Failed to set memory limit" << endl;
#else
        throw SWIGVM::exception("Failed to set memory limit");
#endif
    d.rlim_cur = d.rlim_max = 0;    // 0 for no core dumps, RLIM_INFINITY to enable coredumps of any size
    if (setrlimit(RLIMIT_CORE, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        cerr << "WARNING: Failed to set core limit" << endl;
#else
        throw SWIGVM::exception("Failed to set core limit");
#endif
    /* d.rlim_cur = d.rlim_max = 65536; */
    getrlimit(RLIMIT_NOFILE,&d);
    if (d.rlim_max < 32768)
    {
        //#ifdef SWIGVM_LOG_CLIENT
        cerr << "WARNING: Reducing RLIMIT_NOFILE below 32768" << endl;
        //#endif
    }
    d.rlim_cur = d.rlim_max = std::min(32768,(int)d.rlim_max);
    if (setrlimit(RLIMIT_NOFILE, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        cerr << "WARNING: Failed to set nofile limit" << endl;
#else
        throw SWIGVM::exception("Failed to set nofile limit");
#endif
    d.rlim_cur = d.rlim_max = 32768;
    if (setrlimit(RLIMIT_NPROC, &d) != 0)
    {
        cerr << "WARNING: Failed to set nproc limit to 32k trying 8k ..." << endl;
        d.rlim_cur = d.rlim_max = 8192;
        if (setrlimit(RLIMIT_NPROC, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
            cerr << "WARNING: Failed to set nproc limit" << endl;
#else
            throw SWIGVM::exception("Failed to set nproc limit");
#endif
    }

    { /* send meta request */
        request.Clear();
        request.set_type(MT_META);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer)) {
            exchandler.setException("Communication error: failed to serialize data");
            return false;
        }
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } /* receive meta response */
    {   zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size())) {
            exchandler.setException("Communication error: failed to parse data");
            return false;
        }
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                exchandler.setException(response.close().exception_message().c_str());
            else exchandler.setException("Connection closed by server");
            return false;
        }
        if (response.type() != MT_META) {
            exchandler.setException("Wrong message type, should be META");
            return false;
        }
        const exascript_metadata &rep = response.meta();
        g_singleCallMode = rep.single_call_mode();
        SWIGVM_params_ref->inp_iter_type = (SWIGVM_itertype_e)(rep.input_iter_type());
        SWIGVM_params_ref->out_iter_type = (SWIGVM_itertype_e)(rep.output_iter_type());
        for (int col = 0; col < rep.input_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.input_columns(col);
            SWIGVM_params_ref->inp_names->push_back(coldef.name());
            SWIGVM_params_ref->inp_types->push_back(SWIGVM_columntype_t());
            SWIGVM_columntype_t &coltype = SWIGVM_params_ref->inp_types->back();
            coltype.len = 0; coltype.prec = 0; coltype.scale = 0;
            coltype.type_name = coldef.type_name();
            switch (coldef.type()) {
            case PB_UNSUPPORTED:
                exchandler.setException("Unsupported column type found");
                return false;
            case PB_DOUBLE:
                coltype.type = DOUBLE;
                break;
            case PB_INT32:
                coltype.type = INT32;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_INT64:
                coltype.type = INT64;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_NUMERIC:
                coltype.type = NUMERIC;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_TIMESTAMP:
                coltype.type = TIMESTAMP;
                break;
            case PB_DATE:
                coltype.type = DATE;
                break;
            case PB_STRING:
                coltype.type = STRING;
                coltype.len = coldef.size();
                break;
            case PB_BOOLEAN:
                coltype.type = BOOLEAN;
                break;
            default:
                exchandler.setException("Unknown column type found");
                return false;
            }
        }
        for (int col = 0; col < rep.output_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.output_columns(col);
            SWIGVM_params_ref->out_names->push_back(coldef.name());
            SWIGVM_params_ref->out_types->push_back(SWIGVM_columntype_t());
            SWIGVM_columntype_t &coltype = SWIGVM_params_ref->out_types->back();
            coltype.len = 0; coltype.prec = 0; coltype.scale = 0;
            coltype.type_name = coldef.type_name();
            switch (coldef.type()) {
            case PB_UNSUPPORTED:
                exchandler.setException("Unsupported column type found");
                return false;
            case PB_DOUBLE:
                coltype.type = DOUBLE;
                break;
            case PB_INT32:
                coltype.type = INT32;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_INT64:
                coltype.type = INT64;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_NUMERIC:
                coltype.type = NUMERIC;
                coltype.prec = coldef.precision();
                coltype.scale = coldef.scale();
                break;
            case PB_TIMESTAMP:
                coltype.type = TIMESTAMP;
                break;
            case PB_DATE:
                coltype.type = DATE;
                break;
            case PB_STRING:
                coltype.type = STRING;
                coltype.len = coldef.size();
                break;
            case PB_BOOLEAN:
                coltype.type = BOOLEAN;
                break;
            default:
                exchandler.setException("Unknown column type found");
                return false;
            }
        }
    }
    return true;
}

void send_close(zmq::socket_t &socket, const string &exmsg)
{
    request.Clear();
    request.set_type(MT_CLOSE);
    request.set_connection_id(SWIGVM_params_ref->connection_id);
    exascript_close *req = request.mutable_close();
    if (exmsg != "") req->set_exception_message(exmsg);
    request.SerializeToString(&output_buffer);
    zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
    socket_send(socket, zmsg);

    { /* receive finished response, so we know that the DB knows that we are going to close and
         all potential exceptions have been received on DB side */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if(!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        else if (response.type() != MT_FINISHED)
            throw SWIGVM::exception("Wrong response type, should be finished");
    }
}

bool send_run(zmq::socket_t &socket)
{
    {
        /* send done request */
        request.Clear();
        request.set_type(MT_RUN);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer))
        {
            throw SWIGVM::exception("Communication error: failed to serialize data");
        }
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw SWIGVM::exception(response.close().exception_message().c_str());
            throw SWIGVM::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (g_singleCallMode && response.type() == MT_CALL) {
            assert(g_singleCallMode);
            exascript_single_call_rep sc = response.call();
            g_singleCallFunction = static_cast<single_call_function_id_e>(sc.fn());

            switch (g_singleCallFunction)
            {
            case single_call_function_id_e::SC_FN_NIL:
            case single_call_function_id_e::SC_FN_DEFAULT_OUTPUT_COLUMNS:
                break;
            case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
            {

                if (!sc.has_import_specification())
                {
                    throw SWIGVM::exception("internal error: SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC without import specification");
                }
                const import_specification_rep& is_proto = sc.import_specification();
                g_singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification(is_proto.is_subselect());
                if (is_proto.has_connection_information())
                {
                    const connection_information_rep& ci_proto = is_proto.connection_information();
                    ExecutionGraph::ConnectionInformation connection_info(ci_proto.kind(), ci_proto.address(), ci_proto.user(), ci_proto.password());
                    g_singleCall_ImportSpecificationArg.setConnectionInformation(connection_info);
                }
                if (is_proto.has_connection_name())
                {
                    g_singleCall_ImportSpecificationArg.setConnectionName(is_proto.connection_name());
                }
                for (int i=0; i<is_proto.subselect_column_specification_size(); i++)
                {
                    const ::exascript_metadata_column_definition& cdef = is_proto.subselect_column_specification(i);
                    const ::std::string& cname = cdef.name();
                    const ::std::string& ctype = cdef.type_name();
                    g_singleCall_ImportSpecificationArg.appendSubselectColumnName(cname);
                    g_singleCall_ImportSpecificationArg.appendSubselectColumnType(ctype);
                }
                for (int i=0; i<is_proto.parameters_size(); i++)
                {
                    const ::key_value_pair& kvp = is_proto.parameters(i);
                    g_singleCall_ImportSpecificationArg.addParameter(kvp.key(), kvp.value());
                }
            }
                break;
            case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC:
            {
                if (!sc.has_export_specification())
                {
                    throw SWIGVM::exception("internal error: SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC without export specification");
                }
                const export_specification_rep& es_proto = sc.export_specification();
                g_singleCall_ExportSpecificationArg = ExecutionGraph::ExportSpecification();
                if (es_proto.has_connection_information())
                {
                    const connection_information_rep& ci_proto = es_proto.connection_information();
                    ExecutionGraph::ConnectionInformation connection_info(ci_proto.kind(), ci_proto.address(), ci_proto.user(), ci_proto.password());
                    g_singleCall_ExportSpecificationArg.setConnectionInformation(connection_info);
                }
                if (es_proto.has_connection_name())
                {
                    g_singleCall_ExportSpecificationArg.setConnectionName(es_proto.connection_name());
                }
                for (int i=0; i<es_proto.parameters_size(); i++)
                {
                    const ::key_value_pair& kvp = es_proto.parameters(i);
                    g_singleCall_ExportSpecificationArg.addParameter(kvp.key(), kvp.value());
                }
                g_singleCall_ExportSpecificationArg.setTruncate(es_proto.has_truncate());
                g_singleCall_ExportSpecificationArg.setReplace(es_proto.has_replace());
                if (es_proto.has_created_by())
                {
                    g_singleCall_ExportSpecificationArg.setCreatedBy(es_proto.created_by());
                }
                for (int i=0; i<es_proto.source_column_names_size(); i++)
                {
                    const string name = es_proto.source_column_names(i);
                    g_singleCall_ExportSpecificationArg.addSourceColumnName(name);
                }
            }
                break;
            case single_call_function_id_e::SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                if (!sc.has_json_arg())
                {
                    throw SWIGVM::exception("internal error: SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL without json arg");
                }
                const std::string json = sc.json_arg();
                g_singleCall_StringArg = ExecutionGraph::StringDTO(json);
                break;
            }

            return true;
        } else if (response.type() != MT_RUN) {
            throw SWIGVM::exception("Wrong response type, should be MT_RUN");
        }
    }
    return true;
}

bool send_return(zmq::socket_t &socket, const char* result)
{
  assert(result != nullptr);
    {   /* send return request */
        request.Clear();
        request.set_type(MT_RETURN);
        ::exascript_return_req* rr = new ::exascript_return_req();
        rr->set_result(result);
        request.set_allocated_call_result(rr);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw SWIGVM::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw SWIGVM::exception(response.close().exception_message().c_str());
            throw SWIGVM::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (response.type() != MT_RETURN) {
            throw SWIGVM::exception("Wrong response type, should be MT_RETURN");
        }
    }
    return true;
}

void send_undefined_call(zmq::socket_t &socket, const std::string& fn)
{
    {   /* send return request */
        request.Clear();
        request.set_type(MT_UNDEFINED_CALL);
        ::exascript_undefined_call_req* uc = new ::exascript_undefined_call_req();
        uc->set_remote_fn(fn);
        request.set_allocated_undefined_call(uc);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw SWIGVM::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        if (response.type() != MT_UNDEFINED_CALL) {
            throw SWIGVM::exception("Wrong response type, should be MT_UNDEFINED_CALL");
        }
    }
}


bool send_done(zmq::socket_t &socket)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_DONE);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw SWIGVM::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } 
    { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw SWIGVM::exception(response.close().exception_message().c_str());
            throw SWIGVM::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (response.type() != MT_DONE)
            throw SWIGVM::exception("Wrong response type, should be MT_DONE");
    }
    return true;
}

void send_finished(zmq::socket_t &socket)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_FINISHED);
        request.set_connection_id(SWIGVM_params_ref->connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw SWIGVM::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if(!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVM::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw SWIGVM::exception(response.close().exception_message().c_str());
            throw SWIGVM::exception("Wrong response type, got empty close response");
        } else if (response.type() != MT_FINISHED)
            throw SWIGVM::exception("Wrong response type, should be finished");
    }
}

//
//
// swig log

namespace SWIGVMContainers {

class SWIGMetadata_Impl : public SWIGMetadata {
public:
    SWIGMetadata_Impl():
        SWIGMetadata(false),
        m_connection_id(SWIGVM_params_ref->connection_id),
        m_socket(*(SWIGVM_params_ref->sock)),
        m_exch(SWIGVM_params_ref->exch),
        m_db_name(SWIGVM_params_ref->dbname),
        m_db_version(SWIGVM_params_ref->dbversion),
        m_script_name(SWIGVM_params_ref->script_name),
        m_script_schema(SWIGVM_params_ref->script_schema),
        m_current_user(SWIGVM_params_ref->current_user),
        m_current_schema(SWIGVM_params_ref->current_schema),
        m_scope_user(SWIGVM_params_ref->scope_user),
        m_script_code(SWIGVM_params_ref->script_code),
        m_session_id(SWIGVM_params_ref->session_id),
        m_statement_id(SWIGVM_params_ref->statement_id),
        m_node_count(SWIGVM_params_ref->node_count),
        m_node_id(SWIGVM_params_ref->node_id),
        m_vm_id(SWIGVM_params_ref->vm_id),
        m_input_names(*(SWIGVM_params_ref->inp_names)),
        m_input_types(*(SWIGVM_params_ref->inp_types)),
        m_input_iter_type(SWIGVM_params_ref->inp_iter_type),
        m_output_names(*(SWIGVM_params_ref->out_names)),
        m_output_types(*(SWIGVM_params_ref->out_types)),
        m_output_iter_type(SWIGVM_params_ref->out_iter_type),
        m_memory_limit(SWIGVM_params_ref->maximal_memory_limit),
        m_vm_type(SWIGVM_params_ref->vm_type),
        m_is_emitted(*(SWIGVM_params_ref->is_emitted)),
        m_pluginLanguageName(SWIGVM_params_ref->pluginName),
        m_pluginURI(SWIGVM_params_ref->pluginURI),
        m_outputAddress(SWIGVM_params_ref->outputAddress)
    {
        { std::stringstream sb; sb << m_session_id; m_session_id_s = sb.str(); }
        { std::stringstream sb; sb << m_vm_id; m_vm_id_s = sb.str(); }
    }
    virtual ~SWIGMetadata_Impl() { }
    inline const char* databaseName() { return m_db_name.c_str(); }
    inline const char* databaseVersion() { return m_db_version.c_str(); }
    inline const char* scriptName() { return m_script_name.c_str(); }
    inline const char* scriptSchema() { return m_script_schema.c_str(); }
    inline const char* currentUser() { return m_current_user.c_str(); }
    inline const char* scopeUser() { return m_scope_user.c_str(); }
    inline const char* currentSchema() {return m_current_schema.c_str();}
    inline const char* scriptCode() { return m_script_code.c_str(); }
    inline const unsigned long long sessionID() { return m_session_id; }
    inline const char *sessionID_S() { return m_session_id_s.c_str(); }
    inline const unsigned long statementID() { return m_statement_id; }
    inline const unsigned int nodeCount() { return m_node_count; }
    inline const unsigned int nodeID() { return m_node_id; }
    inline const unsigned long long vmID() { return m_vm_id; }
    inline const unsigned long long memoryLimit() { return m_memory_limit; }
    inline const VMTYPE vmType() { return m_vm_type; }
    inline const char *vmID_S() { return m_vm_id_s.c_str(); }
    inline const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name)
    {
        exascript_request request;
        request.set_type(MT_IMPORT);
        request.set_connection_id(m_connection_id);
        exascript_import_req *req = request.mutable_import();
        req->set_script_name(connection_name);
        req->set_kind(PB_IMPORT_CONNECTION_INFORMATION);
        if (!request.SerializeToString(&m_output_buffer)) {
            m_exch->setException("Communication error: failed to serialize data");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        zmq::message_t zmsg_req((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
        socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("Communication error: failed to parse data");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("Internal error: wrong message type");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            m_exch->setException(rep.exception_message().c_str());
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        if (!rep.has_connection_information()) {
            m_exch->setException("Internal error: No connection information returned");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        connection_information_rep ci = rep.connection_information();
        return new ExecutionGraph::ConnectionInformationWrapper(
                    ExecutionGraph::ConnectionInformation(ci.kind(), ci.address(), ci.user(), ci.password()));
    }
    inline const char* moduleContent(const char* name) {
        exascript_request request;
        request.set_type(MT_IMPORT);
        request.set_connection_id(m_connection_id);
        exascript_import_req *req = request.mutable_import();
        req->set_script_name(name);
        req->set_kind(PB_IMPORT_SCRIPT_CODE);
        if (!request.SerializeToString(&m_output_buffer)) {
            m_exch->setException("Communication error: failed to serialize data");
            return NULL;
        }
        zmq::message_t zmsg_req((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
        socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("Communication error: failed to parse data");
            return NULL;
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("Internal error: wrong message type");
            return NULL;
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            m_exch->setException(rep.exception_message().c_str());
            return NULL;
        }
        if (!rep.has_source_code()) {
            m_exch->setException("Internal error: No source code returned");
            return NULL;
        }
        m_temp_code = rep.source_code();
        return m_temp_code.c_str();
    }
    inline const unsigned int inputColumnCount() { return m_input_names.size(); }
    inline const char *inputColumnName(unsigned int col)
    { return col >= m_input_names.size() ? NULL : m_input_names[col].c_str(); }
    inline const SWIGVM_datatype_e inputColumnType(unsigned int col)
    { return col >= m_input_types.size() ? UNSUPPORTED : m_input_types[col].type; }
    inline const char *inputColumnTypeName(unsigned int col)
    { return col >= m_input_types.size() ? NULL : m_input_types[col].type_name.c_str(); }
    inline const unsigned int inputColumnSize(unsigned int col)
    { return col >= m_input_types.size() ? 0 : m_input_types[col].len; }
    inline const unsigned int inputColumnPrecision(unsigned int col)
    { return col >= m_input_types.size() ? 0 : m_input_types[col].prec; }
    inline const unsigned int inputColumnScale(unsigned int col)
    { return col >= m_input_types.size() ? 0 : m_input_types[col].scale; }
    inline const SWIGVM_itertype_e inputType() { return m_input_iter_type; }
    inline const unsigned int outputColumnCount() { return m_output_names.size(); }
    inline const char *outputColumnName(unsigned int col) {
        if (m_output_iter_type == EXACTLY_ONCE && col == 0)
            return "RETURN";
        return col >= m_output_names.size() ? NULL : m_output_names[col].c_str();
    }
    inline const SWIGVM_datatype_e outputColumnType(unsigned int col)
    { return col >= m_output_types.size() ? UNSUPPORTED : m_output_types[col].type; }
    inline const char *outputColumnTypeName(unsigned int col)
    { return col >= m_output_types.size() ? NULL : m_output_types[col].type_name.c_str(); }
    inline const unsigned int outputColumnSize(unsigned int col)
    { return col >= m_output_types.size() ? 0 : m_output_types[col].len; }
    inline const unsigned int outputColumnPrecision(unsigned int col)
    { return col >= m_output_types.size() ? 0 : m_output_types[col].prec; }
    inline const unsigned int outputColumnScale(unsigned int col)
    { return col >= m_output_types.size() ? 0 : m_output_types[col].scale; }
    inline const SWIGVM_itertype_e outputType() { return m_output_iter_type; }
    inline const bool isEmittedColumn(unsigned int col){
        if (col >= m_is_emitted.size())
        {
            abort();
        }
        return m_is_emitted[col];
    }
    inline const char* checkException() {
        if (m_exch->exthrowed) {
            m_exch->exthrowed = false;
            return m_exch->exmsg.c_str();
        } else return NULL;
    }
    inline const char* pluginLanguageName()
    {
        return m_pluginLanguageName.c_str();
    }
    inline const char* pluginURI()
    {
        return m_pluginURI.c_str();
    }
    inline const char* outputAddress()
    {
        return m_outputAddress.c_str();
    }
private:
    const uint64_t m_connection_id;
    zmq::socket_t &m_socket;
    std::string m_output_buffer;
    SWIGVMExceptionHandler *m_exch;
    const std::string m_db_name;
    const std::string m_db_version;
    const std::string m_script_name;
    const std::string m_script_schema;
    const std::string m_current_user;
    const std::string m_current_schema;
    const std::string m_scope_user;
    const std::string m_script_code;
    const unsigned long m_session_id;
    const unsigned long m_statement_id;
    const unsigned int m_node_count;
    const unsigned int m_node_id;
    const unsigned long m_vm_id;
    std::string m_temp_code;
    const std::vector<std::string> &m_input_names;
    const std::vector<SWIGVM_columntype_t> &m_input_types;
    const SWIGVM_itertype_e m_input_iter_type;
    const std::vector<std::string> &m_output_names;
    const std::vector<SWIGVM_columntype_t> &m_output_types;
    const SWIGVM_itertype_e m_output_iter_type;
    const unsigned long long m_memory_limit;
    const std::string m_meta_info;
    const VMTYPE m_vm_type;
    std::string m_session_id_s;
    std::string m_vm_id_s;
    const std::vector<bool> &m_is_emitted;
    std::string m_pluginLanguageName;
    std::string m_pluginURI;
    std::string m_outputAddress;
};



class SWIGGeneralIterator {
    protected:
        SWIGVMExceptionHandler *m_exch;
    public:
//        SWIGGeneralIterator(SWIGVMExceptionHandler *exch): m_exch(exch) { }
        SWIGGeneralIterator()
            : m_exch(SWIGVM_params_ref->exch)
        {}
        virtual ~SWIGGeneralIterator() { }
        inline const char* checkException() {
            if (m_exch->exthrowed) {
                m_exch->exthrowed = false;
                return m_exch->exmsg.c_str();
            } else return NULL;
        }
};


class SWIGTableIterator_Impl : public AbstractSWIGTableIterator, SWIGGeneralIterator {
private:
    const uint64_t m_connection_id;
    zmq::socket_t &m_socket;
    std::string m_output_buffer;
    exascript_request m_request;
    exascript_response m_next_response;

    uint64_t m_rows_received;
    struct values_per_row_t {
        uint64_t strings, bools, int32s, int64s, doubles;
        values_per_row_t(): strings(0), bools(0), int32s(0), int64s(0), doubles(0) {}
        void reset() { strings = bools = int32s = int64s = doubles = 0; }
    } m_values_per_row;
    uint64_t m_column_count;
    std::vector<uint64_t> m_col_offsets;
    uint64_t m_rows_in_group;
    uint64_t m_current_row;
    uint64_t m_rows_completed;
    uint64_t m_rows_group_completed;
    bool m_was_null;
    const std::vector<SWIGVM_columntype_t> &m_types;
    void increment_col_offsets(bool reset = false) {
        m_current_row = m_next_response.next().table().row_number(m_rows_completed);
        uint64_t current_column = 0;
        ssize_t null_index = 0;
        if (reset) m_values_per_row.reset();
        null_index = m_rows_completed * m_column_count;
        if (m_next_response.next().table().data_nulls_size() <= (null_index + (ssize_t)m_column_count - 1)) {
            std::stringstream sb;
            sb << "Internal error: not enough nulls in packet: wanted index " << (null_index + m_column_count - 1)
               << " but have " << m_next_response.next().table().data_nulls_size()
               << " elements";
            m_exch->setException(sb.str().c_str());
            return;
        }
        for (std::vector<SWIGVM_columntype_t>::const_iterator
             it = m_types.begin(); it != m_types.end(); ++it, ++current_column, ++null_index)
        {
            if (m_next_response.next().table().data_nulls(null_index))
                continue;
            switch (it->type) {
            case UNSUPPORTED: m_exch->setException("Unsupported data type found"); return;
            case DOUBLE: m_col_offsets[current_column] = m_values_per_row.doubles++; break;
            case INT32: m_col_offsets[current_column] = m_values_per_row.int32s++; break;
            case INT64: m_col_offsets[current_column] = m_values_per_row.int64s++; break;
            case NUMERIC:
            case TIMESTAMP:
            case DATE:
            case STRING: m_col_offsets[current_column] = m_values_per_row.strings++; break;
            case BOOLEAN: m_col_offsets[current_column] = m_values_per_row.bools++; break;
            default: m_exch->setException("Unknown data type found"); return;
            }
        }
    }
    void receive_next_data(bool reset) {
        m_rows_received = 0;
        m_rows_completed = 0;
        {
            m_request.Clear();
            m_request.set_connection_id(m_connection_id);
            if (reset) m_request.set_type(MT_RESET);
            else m_request.set_type(MT_NEXT);
            if(!m_request.SerializeToString(&m_output_buffer)) {
                m_exch->setException("Communication error: failed to serialize data");
                return;
            }
            zmq::message_t zmsg((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
            socket_send(m_socket, zmsg);
        } {
            zmq::message_t zmsg;
            socket_recv(m_socket, zmsg);
            m_next_response.Clear();
            if (!m_next_response.ParseFromArray(zmsg.data(), zmsg.size())) {
                m_exch->setException("Communication error: failed to parse data");
                return;
            }
            if (m_next_response.connection_id() != m_connection_id) {
                m_exch->setException("Communication error: wrong connection id");
                return;
            }
            if (m_next_response.type() == MT_DONE) {
                return;
            }
            if (m_next_response.type() == MT_CLOSE) {
                const exascript_close &rep = m_next_response.close();
                if (!rep.has_exception_message()) {
                    if (m_rows_completed == 0) {
                        return;
                    } else m_exch->setException("Unknown error occured");
                } else {
                    m_exch->setException(rep.exception_message().c_str());
                }
                return;
            }
            if ((reset && (m_next_response.type() != MT_RESET)) ||
                    (!reset && (m_next_response.type() != MT_NEXT)))
            {
                m_exch->setException("Communication error: wrong message type");
                return;
            }
            m_rows_received = m_next_response.next().table().rows();
            m_rows_in_group = m_next_response.next().table().rows_in_group();
        }
        increment_col_offsets(true);
    }
    inline ssize_t check_index(ssize_t index, ssize_t available, const char *tp, const char *otype, const char *ts) {
        if (available > index) return index;
        std::stringstream sb;
        sb << "Internal error: not enough " << tp << otype << ts << " in packet: wanted index "
           << index << " but have " << available << " elements (on "
           << m_rows_received << '/' << m_rows_completed << " of received/completed rows";
        m_exch->setException(sb.str().c_str());
        m_was_null = true;
        return -1;
    }
    inline ssize_t check_value(unsigned int col, ssize_t available, const char *otype) {
        m_was_null = false;
        ssize_t index = check_index(m_column_count * m_rows_completed + col,
                                    m_next_response.next().table().data_nulls_size(),
                                    "nulls for ", otype, "");
        if (m_was_null) return -1;
        m_was_null = m_next_response.next().table().data_nulls(index);
        if (m_was_null) return -1;
        index = check_index(m_col_offsets[col], available, "", otype, "s");
        if (m_was_null) return -1;
        return index;
    }
public:
    const char* checkException() {return SWIGGeneralIterator::checkException();}
    SWIGTableIterator_Impl():        
        m_connection_id(SWIGVM_params_ref->connection_id),
        m_socket(*(SWIGVM_params_ref->sock)),
        m_column_count(SWIGVM_params_ref->inp_types->size()),
        m_col_offsets(SWIGVM_params_ref->inp_types->size()),
        m_current_row((uint64_t)-1),
        m_rows_completed(0),
        m_rows_group_completed(1),
        m_was_null(false),
        m_types(*(SWIGVM_params_ref->inp_types))
    {
        receive_next_data(false);
    }
    ~SWIGTableIterator_Impl() {
    }
    uint64_t get_current_row()
    {
        return m_current_row;
    }
    inline void reinitialize() {
        m_rows_completed = 0;
        m_rows_group_completed = 1;
        m_values_per_row.reset();
        receive_next_data(false);
    }
    inline bool next() {
        if (m_rows_received == 0) {
            m_exch->setException("Iteration finished");
            return false;
        }
        ++m_rows_completed;
        ++m_rows_group_completed;
        if (SWIGVM_params_ref->inp_force_finish)
            return false;
        if (m_rows_completed >= m_rows_received) {
            receive_next_data(false);
            if (m_rows_received == 0)
                return false;
            else return true;
        }
        increment_col_offsets();
        return true;
    }
    inline bool eot() {
        return m_rows_received == 0;
    }
    inline void reset() {
        m_rows_group_completed = 1;
        SWIGVM_params_ref->inp_force_finish = false;
        receive_next_data(true);
    }
    inline unsigned long restBufferSize() {
        if (m_rows_completed < m_rows_received)
            return m_rows_received - m_rows_completed;
        return 0;
    }
    inline unsigned long rowsCompleted() {
        return m_rows_group_completed;
    }
    inline unsigned long rowsInGroup() {
        return m_rows_in_group;
    }
    inline double getDouble(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0.0; }
        if (m_types[col].type != DOUBLE) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return 0.0;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_double_size(), "double");
        if (m_was_null) {
            return 0.0;
        }
        return m_next_response.next().table().data_double(index);
    }
    inline const char *getString(unsigned int col, size_t *length = NULL) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (m_types[col].type != STRING) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return "";
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "";
        const std::string &s(m_next_response.next().table().data_string(index));
        if (length != NULL) *length = s.length();
        return s.c_str();
    }
    inline int32_t getInt32(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0; }
        if (m_types[col].type != INT32) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return 0;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int32_size(), "int32");
        if (m_was_null) return 0;
        return m_next_response.next().table().data_int32(index);
    }
    inline int64_t getInt64(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0LL; }
        if (m_types[col].type != INT64) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return 0LL;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int64_size(), "int64");
        if (m_was_null) return 0LL;
        return m_next_response.next().table().data_int64(index);
    }
    inline const char *getNumeric(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (m_types[col].type != NUMERIC) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "0";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getTimestamp(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (m_types[col].type != TIMESTAMP) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01 00:00:00.00 0000";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getDate(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (m_types[col].type != DATE) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline bool getBoolean(unsigned int col) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (m_types[col].type != BOOLEAN) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_bool_size(), "bool");
        if (m_was_null) return false;
        return m_next_response.next().table().data_bool(index);
    }
    inline bool wasNull() { return m_was_null; }
};

class SWIGResultHandler_Impl : public SWIGRAbstractResultHandler, SWIGGeneralIterator {
private:
    SWIGTableIterator* m_table_iterator;
    const uint64_t m_connection_id;
    zmq::socket_t &m_socket;
    std::string m_output_buffer;
    uint64_t m_message_size;
    struct rowdata_t {
        std::map<int, bool> null_data;
        std::map<int, bool> bool_data;
        std::map<int, double> double_data;
        std::map<int, int32_t> int32_data;
        std::map<int, int64_t> int64_data;
        std::map<int, std::string> string_data;
    };
    rowdata_t m_rowdata;
    exascript_request m_emit_request;
    uint64_t m_rows_emited;
    const std::vector<SWIGVM_columntype_t> &m_types;
public:
    const char* checkException() {return SWIGGeneralIterator::checkException();}
    SWIGResultHandler_Impl(SWIGTableIterator* table_iterator)
        : m_table_iterator(table_iterator),
          m_connection_id(SWIGVM_params_ref->connection_id),
          m_socket(*(SWIGVM_params_ref->sock)),
          m_message_size(0),
          m_rows_emited(1),
          m_types(*(SWIGVM_params_ref->out_types))
    { }
    ~SWIGResultHandler_Impl() {
    }
    inline void reinitialize() {
        m_rows_emited = 0;
        m_message_size = 0;
        m_emit_request.Clear();
        m_rowdata = rowdata_t();
    }
    inline unsigned long rowsEmited() {
        return m_rows_emited;
    }
    inline void flush() {
        exascript_emit_data_req *req = m_emit_request.mutable_emit();
        exascript_table_data *table = req->mutable_table();
        if (table->has_rows() && table->rows() > 0) {
            { m_emit_request.set_type(MT_EMIT);
                m_emit_request.set_connection_id(m_connection_id);
                if (!m_emit_request.SerializeToString(&m_output_buffer)) {
                    m_exch->setException("Communication error: failed to serialize data");
                    return;
                }
                zmq::message_t zmsg((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
                socket_send(m_socket, zmsg);
                m_emit_request.Clear();
                m_message_size = 0;
            }
            { zmq::message_t zmsg;
                socket_recv(m_socket, zmsg);
                exascript_response response;
                if (!response.ParseFromArray(zmsg.data(), zmsg.size())) {
                    m_exch->setException("Communication error: failed to parse data");
                    return;
                }
                if (response.connection_id() != m_connection_id) {
                    std::stringstream sb;
                    sb << "Received wrong connection id " << response.connection_id()
                       << ", should be " << m_connection_id;
                    m_exch->setException(sb.str().c_str());
                    return;
                }
                if (response.type() == MT_CLOSE) {
                    if (!response.close().has_exception_message())
                        m_exch->setException("Unknown error occured");
                    else m_exch->setException(response.close().exception_message().c_str());
                    return;
                }
                if (response.type() != MT_EMIT) {
                    m_exch->setException("Wrong response type");
                    return;
                }
            }
        }
    }
    inline bool next() {
        ++m_rows_emited;
        exascript_emit_data_req *req = m_emit_request.mutable_emit();
        exascript_table_data *table = req->mutable_table();
        for (unsigned int col = 0; col < m_types.size(); ++col) {
            bool null_data = m_rowdata.null_data[col];
            table->add_data_nulls(null_data);
            if (null_data) continue;
            switch (m_types[col].type) {
            case UNSUPPORTED:
                m_exch->setException("Unsupported data type found");
                return false;
            case DOUBLE:
                if (m_rowdata.double_data.find(col) == m_rowdata.double_data.end()) {
                    m_exch->setException("Not enough double columns emited");
                    return false;
                }
                m_message_size += sizeof(double);
                table->add_data_double(m_rowdata.double_data[col]);
                break;
            case INT32:
                if (m_rowdata.int32_data.find(col) == m_rowdata.int32_data.end()) {
                    m_exch->setException("Not enough int32 columns emited");
                    return false;
                }
                m_message_size += sizeof(int32_t);
                table->add_data_int32(m_rowdata.int32_data[col]);
                break;
            case INT64:
                if (m_rowdata.int64_data.find(col) == m_rowdata.int64_data.end()) {
                    m_exch->setException("Not enough int64 columns emited");
                    return false;
                }
                m_message_size += sizeof(int64_t);
                table->add_data_int64(m_rowdata.int64_data[col]);
                break;
            case NUMERIC:
            case TIMESTAMP:
            case DATE:
            case STRING:
                if (m_rowdata.string_data.find(col) == m_rowdata.string_data.end()) {
                    m_exch->setException("Not enough string columns emited");
                    return false;
                }
                m_message_size += sizeof(int32_t) + m_rowdata.string_data[col].length();
                *table->add_data_string() = m_rowdata.string_data[col];
                break;
            case BOOLEAN:
                if (m_rowdata.bool_data.find(col) == m_rowdata.bool_data.end()) {
                    m_exch->setException("Not enough boolean columns emited");
                    return false;
                }
                m_message_size += 1;
                table->add_data_bool(m_rowdata.bool_data[col]);
                break;
            default:
                m_exch->setException("Unknown data type found");
                return false;
            }
        }
        table->add_row_number(m_table_iterator->get_current_row());
        m_rowdata = rowdata_t();
        if (!table->has_rows()) table->set_rows(1);
        else table->set_rows(table->rows() + 1);
        table->set_rows_in_group(0);
        if (m_message_size >= SWIG_MAX_VAR_DATASIZE) {
            if (SWIGVM_params_ref->inp_iter_type == EXACTLY_ONCE && SWIGVM_params_ref->out_iter_type == EXACTLY_ONCE)
                SWIGVM_params_ref->inp_force_finish = true;
            else this->flush();
        }
        return true;
    }
    inline void setDouble(unsigned int col, const double v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != DOUBLE) { m_exch->setException("Wrong column type (not a double)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.double_data[col] = v;
    }
    inline void setString(unsigned int col, const char *v, size_t l) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != STRING) { m_exch->setException("Wrong column type (not a string)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setInt32(unsigned int col, const int32_t v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != INT32) { m_exch->setException("Wrong column type (not Int32)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.int32_data[col] = v;
    }
    inline void setInt64(unsigned int col, const int64_t v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != INT64) { m_exch->setException("Wrong column type (not Int64)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.int64_data[col] = v;
    }
    inline void setNumeric(unsigned int col, const char *v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != NUMERIC) { m_exch->setException("Wrong column type (not Numeric)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setTimestamp(unsigned int col, const char *v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != TIMESTAMP) { m_exch->setException("Wrong column type (not Timestamp)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setDate(unsigned int col, const char *v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != DATE) { m_exch->setException("Wrong column type (not Date)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setBoolean(unsigned int col, const bool v) {
        if (col >= m_types.size()) { m_exch->setException("Column does not exist"); return; }
        if (m_types[col].type != BOOLEAN) { m_exch->setException("Wrong column type (not Boolean)"); return; }
        m_rowdata.null_data[col] = false;
        m_rowdata.bool_data[col] = v;
    }
    inline void setNull(unsigned int col) {
        m_rowdata.null_data[col] = true;
    }
};




} // namespace SWIGVMContainers

extern "C" {

SWIGVMContainers::SWIGMetadata* create_SWIGMetaData() {
    return new SWIGVMContainers::SWIGMetadata_Impl();
}

SWIGVMContainers::AbstractSWIGTableIterator* create_SWIGTableIterator() {
    return new SWIGVMContainers::SWIGTableIterator_Impl();
}

SWIGVMContainers::SWIGRAbstractResultHandler* create_SWIGResultHandler(SWIGVMContainers::SWIGTableIterator* table_iterator) {
    return new SWIGVMContainers::SWIGResultHandler_Impl(table_iterator);
}



int exaudfclient_main(std::function<SWIGVM*()>vmMaker,int argc,char**argv)
{
    assert(SWIGVM_params_ref != nullptr);

#ifdef PROTEGRITY_PLUGIN_CLIENT
    stringstream socket_name_ss;
#endif
    string socket_name = argv[1];
    char* socket_name_str = argv[1];
    const char *socket_name_file = argv[1];

    init_socket_name(socket_name_str);

    set_remote_client(false);
    my_pid = ::getpid();

    zmq::context_t context(1);

#ifdef SWIGVM_LOG_CLIENT
    for (int i = 0; i<argc; i++)
    {
        cerr << "zmqcontainerclient argv[" << i << "] = " << argv[i] << endl;
    }
#endif

    if (socket_name.length() > 4 ) {
#ifdef PROTEGRITY_PLUGIN_CLIENT
        // protegrity client has no arguments
#else
        if (! ((strcmp(argv[2], "lang=python") == 0)
               || (strcmp(argv[2], "lang=r") == 0)
               || (strcmp(argv[2], "lang=java") == 0)
               || (strcmp(argv[2], "lang=streaming") == 0)) )
        {
            cerr << "Remote VM type '" << argv[3] << "' not supported." << endl;
            return 2;
        }
#endif
    } else {
        abort();
    }

    if (strncmp(socket_name_str, "tcp:", 4) == 0) {
        set_remote_client(true);
    }

    if (socket_name.length() > 6 && strncmp(socket_name_str, "ipc:", 4) == 0)
    {
#ifdef PROTEGRITY_PLUGIN_CLIENT
        if (strncmp(socket_name_str, "ipc:///tmp/", 11) == 0) {
            socket_name_ss << "ipc://" << getenv("NSEXEC_TMP_PATH") << '/' << &(socket_name_file[11]);
            socket_name = socket_name_ss.str();
            socket_name_str = strdup(socket_name_ss.str().c_str());
            socket_name_file = socket_name_str;
        }
#endif
        socket_name_file = &(socket_name_file[6]);
    }

#ifdef SWIGVM_LOG_CLIENT
    cerr << "### SWIGVM starting " << argv[0] << " with name '" << socket_name
         << " (" << ::getppid() << ',' << ::getpid() << "): '"
         << argv[1]
         << '\'' << endl;
#endif

    start_check_thread();


    int linger_timeout = 0;
    int recv_sock_timeout = 1000;
    int send_sock_timeout = 1000;

    if (get_remote_client()) {
        recv_sock_timeout = 10000;
        send_sock_timeout = 5000;
    }

reinit:
    zmq::socket_t socket(context, ZMQ_REQ);

    socket.setsockopt(ZMQ_LINGER, &linger_timeout, sizeof(linger_timeout));
    socket.setsockopt(ZMQ_RCVTIMEO, &recv_sock_timeout, sizeof(recv_sock_timeout));
    socket.setsockopt(ZMQ_SNDTIMEO, &send_sock_timeout, sizeof(send_sock_timeout));

    if (get_remote_client()) socket.bind(socket_name_str);
    else socket.connect(socket_name_str);

    SWIGVM_params_ref->sock = &socket;
    SWIGVM_params_ref->exch = &exchandler;

    if (!send_init(socket, socket_name)) {
        if (!get_remote_client() && exchandler.exthrowed) {
            send_close(socket, exchandler.exmsg);
            return 1;
        }
        goto reinit;
    }

    SWIGVM_params_ref->dbname = (char*) g_database_name.c_str();
    SWIGVM_params_ref->dbversion = (char*) g_database_version.c_str();
    SWIGVM_params_ref->script_name = (char*) g_script_name.c_str();
    SWIGVM_params_ref->script_schema = (char*) g_script_schema.c_str();
    SWIGVM_params_ref->current_user = (char*) g_current_user.c_str();
    SWIGVM_params_ref->current_schema = (char*) g_current_schema.c_str();
    SWIGVM_params_ref->scope_user = (char*) g_scope_user.c_str();
    SWIGVM_params_ref->script_code = (char*) g_source_code.c_str();
    SWIGVM_params_ref->session_id = g_session_id;
    SWIGVM_params_ref->statement_id = g_statement_id;
    SWIGVM_params_ref->node_count = g_node_count;
    SWIGVM_params_ref->node_id = g_node_id;
    SWIGVM_params_ref->vm_id = g_vm_id;
    SWIGVM_params_ref->singleCallMode = g_singleCallMode;

    SWIGVM*vm=nullptr;

    try {
        vm = vmMaker();
        if (vm == nullptr) {
            send_close(socket, "Unknown or unsupported VM type");
            return 1;
        }
        if (vm->exception_msg.size()>0) {
            throw SWIGVM::exception(vm->exception_msg.c_str());
        }

        use_zmq_socket_locks = vm->useZmqSocketLocks();

        if (g_singleCallMode) {
            ExecutionGraph::EmptyDTO noArg; // used as dummy arg
            for (;;) {
                // in single call mode, after MT_RUN from the client,
                // EXASolution responds with a CALL message that specifies
                // the single call function to be made
                if (!send_run(socket)) {
		  break;
		}
                assert(g_singleCallFunction != single_call_function_id_e::SC_FN_NIL);
		try {
		    const char* result = nullptr;
                    switch (g_singleCallFunction)
                    {
                    case single_call_function_id_e::SC_FN_NIL:
                        break;
                    case single_call_function_id_e::SC_FN_DEFAULT_OUTPUT_COLUMNS:
		        result = vm->singleCall(g_singleCallFunction,noArg);
                        break;
                    case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
                        assert(!g_singleCall_ImportSpecificationArg.isEmpty());
                        result = vm->singleCall(g_singleCallFunction,g_singleCall_ImportSpecificationArg);
                        g_singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification();  // delete the last argument
                        break;
                    case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC:
                        assert(!g_singleCall_ExportSpecificationArg.isEmpty());
                        result = vm->singleCall(g_singleCallFunction,g_singleCall_ExportSpecificationArg);
                        g_singleCall_ExportSpecificationArg = ExecutionGraph::ExportSpecification();  // delete the last argument
                        break;
                    case single_call_function_id_e::SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                        assert(!g_singleCall_StringArg.isEmpty());
			result = vm->singleCall(g_singleCallFunction,g_singleCall_StringArg);
			break;
                    }
                    if (vm->exception_msg.size()>0) {
                        send_close(socket, vm->exception_msg); socket.close();
                        goto error;
                    }

                    if (vm->calledUndefinedSingleCall.size()>0) {
                         send_undefined_call(socket, vm->calledUndefinedSingleCall);
                     } else {
		       send_return(socket,result);
                     }

                    if (!send_done(socket)) {
                        break;
                    }
		} catch(...) {
		}
            }
        } else {
            for(;;) {
                if (!send_run(socket))
                    break;
                SWIGVM_params_ref->inp_force_finish = false;
                while(!vm->run_())
                {
                    if (vm->exception_msg.size()>0) {
                        send_close(socket, vm->exception_msg); socket.close();
                        goto error;
                    }
                }
                if (!send_done(socket))
                    break;
            }
        }

        if (vm != nullptr)
        {
            vm->shutdown();
            if (vm->exception_msg.size()>0) {
                send_close(socket, vm->exception_msg); socket.close();
                goto error;
            }
            delete vm;
            vm = NULL;
        }
        send_finished(socket);
    }  catch (SWIGVM::exception &err) {
        keep_checking = false;
        send_close(socket, err.what()); socket.close();

#ifdef SWIGVM_LOG_CLIENT
        cerr << "### SWIGVM crashing with name '" << socket_name
             << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what() << endl;
#endif
        goto error;
    } catch (std::exception &err) {
        send_close(socket, err.what()); socket.close();
#ifdef SWIGVM_LOG_CLIENT
        cerr << "### SWIGVM crashing with name '" << socket_name
             << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what() << endl;
#endif
        goto error;
    } catch (...) {
        send_close(socket, "Internal/Unknown error"); socket.close();
#ifdef SWIGVM_LOG_CLIENT
        cerr << "### SWIGVM crashing with name '" << socket_name
             << " (" << ::getppid() << ',' << ::getpid() << ')' << endl;
#endif
        goto error;
    }

#ifdef SWIGVM_LOG_CLIENT
    cerr << "### SWIGVM finishing with name '" << socket_name
         << " (" << ::getppid() << ',' << ::getpid() << ')' << endl;
#endif
    stop_check_thread();
    socket.close();
    if (!get_remote_client()) {
        cancel_check_thread();
        ::unlink(socket_name_file);
    }
    return 0;

error:
    keep_checking = false;
    if (vm != NULL)
    {
        vm->shutdown();
        delete vm;
        vm = NULL;
    }

    socket.close();
    if (!get_remote_client()) {
        cancel_check_thread();
        ::unlink(socket_name_file);
    } else {
        ::sleep(3); // give other components time to shutdown
    }
    return 1;
}


} // extern "C"
