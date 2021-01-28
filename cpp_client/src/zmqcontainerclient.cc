#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <swigcontainers_ext.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include <fcntl.h>
#include <fstream>
#include "scriptDTO.h"

#ifdef ENABLE_CPP_VM
#include "cpp.h"
#endif

#ifdef PROTEGRITY_PLUGIN_CLIENT
#include <protegrityclient.h>
#endif

using namespace SWIGVMContainers;
using namespace std;
using namespace google::protobuf;

__thread SWIGVM_params_t *SWIGVMContainers::SWIGVM_params;

static string socket_name;
static const char *socket_name_str;
static string output_buffer;
static SWIGVMExceptionHandler exchandler;
static pid_t my_pid; //parent_pid,
//static exascript_vmtype vm_type;
static exascript_request request;
static exascript_response response;

static string g_database_name;
static string g_database_version;
static string g_script_name;
static string g_script_schema;
static string g_current_user;
static string g_current_schema;
static string g_source_code;
static unsigned long long g_session_id;
static unsigned long g_statement_id;
static unsigned int g_node_count;
static unsigned int g_node_id;
static unsigned long long g_vm_id;
static bool g_singleCallMode;
static single_call_function_id g_singleCallFunction;
static ExecutionGraph::ImportSpecification g_singleCall_ImportSpecificationArg;
static ExecutionGraph::StringDTO g_singleCall_StringArg;
static bool remote_client;

#ifndef NDEBUG
#define SWIGVM_LOG_CLIENT
#endif
#define SWIGVM_LOG_CLIENT
//#define LOG_COMMUNICATION

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


static bool keep_checking = true;

void *check_thread_routine(void* data)
{
    while(keep_checking) {
        external_process_check();
        ::usleep(100000);
    }
    return NULL;

}

void SWIGVMContainers::socket_send(zmq::socket_t &socket, zmq::message_t &zmsg)
{
#ifdef LOG_COMMUNICATION
    stringstream sb;
    uint32_t len = zmsg.size();
    sb << "/tmp/zmqcomm_log_" << ::getpid() << "_send.data";
    int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
    if (fd >= 0) {
        ::write(fd, &len, sizeof(uint32_t));
        ::write(fd, zmsg.data(), len);
        ::close(fd);
    }
#endif
    for (;;) {
        try {
            if (socket.send(zmsg) == true)
                return;
            external_process_check();
        } catch (std::exception &err) {
            external_process_check();
        } catch (...) {
            external_process_check();
        }
        ::usleep(100000);
    }
}

bool SWIGVMContainers::socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error)
{
    for (;;) {
        try {
            if (socket.recv(&zmsg) == true) {
#ifdef LOG_COMMUNICATION
                stringstream sb;
                uint32_t len = zmsg.size();
                sb << "/tmp/zmqcomm_log_" << ::getpid() << "_recv.data";
                int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
                if (fd >= 0) {
                    ::write(fd, &len, sizeof(uint32_t));
                    ::write(fd, zmsg.data(), len);
                    ::close(fd);
                }
#endif
                return true;
            }
            external_process_check();
        } catch (std::exception &err) {
            external_process_check();
            
        } catch (...) {
            external_process_check();
        }
        if (return_on_error) return false;
        ::usleep(100000);
    }
    return false;
}

static bool send_init(zmq::socket_t &socket, const string client_name)
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

    SWIGVM_params->connection_id = response.connection_id();
#ifdef SWIGVM_LOG_CLIENT
    stringstream sb; sb << std::hex << SWIGVM_params->connection_id;
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
    g_current_schema = rep.current_schema();
    g_source_code = rep.source_code();
    g_session_id = rep.session_id();
    g_statement_id = rep.statement_id();
    g_node_count = rep.node_count();
    g_node_id = rep.node_id();
    g_vm_id = rep.vm_id();
    //vm_type = rep.vm_type();


    SWIGVM_params->maximal_memory_limit = rep.maximal_memory_limit();
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
        request.set_connection_id(SWIGVM_params->connection_id);
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
        SWIGVM_params->inp_iter_type = (SWIGVM_itertype_e)(rep.input_iter_type());
        SWIGVM_params->out_iter_type = (SWIGVM_itertype_e)(rep.output_iter_type());
        for (int col = 0; col < rep.input_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.input_columns(col);
            SWIGVM_params->inp_names->push_back(coldef.name());
            SWIGVM_params->inp_types->push_back(SWIGVM_columntype_t());
            SWIGVM_columntype_t &coltype = SWIGVM_params->inp_types->back();
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
            SWIGVM_params->out_names->push_back(coldef.name());
            SWIGVM_params->out_types->push_back(SWIGVM_columntype_t());
            SWIGVM_columntype_t &coltype = SWIGVM_params->out_types->back();
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

static void send_close(zmq::socket_t &socket, const string &exmsg)
{
    request.Clear();
    request.set_type(MT_CLOSE);
    request.set_connection_id(SWIGVM_params->connection_id);
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

static bool send_run(zmq::socket_t &socket)
{
    {
        /* send done request */
        request.Clear();
        request.set_type(MT_RUN);
        request.set_connection_id(SWIGVM_params->connection_id);
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
            g_singleCallFunction = sc.fn();

            switch (g_singleCallFunction)
            {
            case SC_FN_NIL:
            case SC_FN_DEFAULT_OUTPUT_COLUMNS:
                break;
            case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
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
            case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                // TODO VS This will be refactored soon, just temporary
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
            throw SWIGVM::exception("Wrong response type, should be done");
        }
    }
    return true;
}


static bool send_return(zmq::socket_t &socket, std::string& result)
{
    {   /* send return request */
        request.Clear();
        request.set_type(MT_RETURN);
        ::exascript_return_req* rr = new ::exascript_return_req();
        rr->set_result(result.c_str());
        request.set_allocated_call_result(rr);
        request.set_connection_id(SWIGVM_params->connection_id);
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

static void send_undefined_call(zmq::socket_t &socket, const std::string& fn)
{
    {   /* send return request */
        request.Clear();
        request.set_type(MT_UNDEFINED_CALL);
        ::exascript_undefined_call_req* uc = new ::exascript_undefined_call_req();
        uc->set_remote_fn(fn);
        request.set_allocated_undefined_call(uc);
        request.set_connection_id(SWIGVM_params->connection_id);
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


static bool send_done(zmq::socket_t &socket)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_DONE);
        request.set_connection_id(SWIGVM_params->connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw SWIGVM::exception("Communication error: failed to serialize data");
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
        } else if (response.type() != MT_DONE)
            throw SWIGVM::exception("Wrong response type, should be done");
    }
    return true;
}

static void send_finished(zmq::socket_t &socket)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_FINISHED);
        request.set_connection_id(SWIGVM_params->connection_id);
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

int main(int argc, char **argv) {
#ifdef PROTEGRITY_PLUGIN_CLIENT
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <socket>" << endl;
        return 1;
    }
#else
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <socket> lang=python|lang=r|lang=java" << endl;
        return 1;
    }
#endif

    if (::setenv("HOME", "/tmp", 1) == -1)
    {
        throw SWIGVM::exception("Failed to set HOME directory");
    }
    ::setlocale(LC_ALL, "en_US.utf8");

    
#ifdef PROTEGRITY_PLUGIN_CLIENT
    stringstream socket_name_ss;
#endif
    socket_name = argv[1];
    socket_name_str = argv[1];
    const char *socket_name_file = argv[1];

    remote_client = false;
    my_pid = ::getpid();
    SWIGVM_params = new SWIGVM_params_t(true);
    zmq::context_t context(1);

#ifdef SWIGVM_LOG_CLIENT
    for (int i = 0; i<argc; i++)
    {
        cerr << "zmqcontainerclient argv[" << i << "] = " << argv[i] << endl;
    }
#endif

    if (socket_name.length() > 4 ) {
#ifdef PROTEGRITY_PLUGIN_CLIENT
        // udf plugins might not have arguments
#else
//        if (! ((strcmp(argv[2], "lang=python") == 0)
//               || (strcmp(argv[2], "lang=r") == 0)
//               || (strcmp(argv[2], "lang=java") == 0)) )
//        {
//            cerr << "Remote VM type '" << argv[3] << "' not supported." << endl;
//            return 2;
//        }
#endif
    } else {
        abort();
    }

    if (strncmp(socket_name_str, "tcp:", 4) == 0) {
            remote_client = true;
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
    pthread_t check_thread;
    if (!remote_client)
        pthread_create(&check_thread, NULL, check_thread_routine, NULL);

    int linger_timeout = 0;
    int recv_sock_timeout = 1000;
    int send_sock_timeout = 1000;

    if (remote_client) {
        recv_sock_timeout = 10000;
        send_sock_timeout = 5000;
    }

reinit:
    zmq::socket_t socket(context, ZMQ_REQ);

    socket.setsockopt(ZMQ_LINGER, &linger_timeout, sizeof(linger_timeout));
    socket.setsockopt(ZMQ_RCVTIMEO, &recv_sock_timeout, sizeof(recv_sock_timeout));
    socket.setsockopt(ZMQ_SNDTIMEO, &send_sock_timeout, sizeof(send_sock_timeout));

    if (remote_client) socket.bind(socket_name_str);
    else socket.connect(socket_name_str);

    SWIGVM_params->sock = &socket;
    SWIGVM_params->exch = &exchandler;

    if (!send_init(socket, socket_name)) {
        if (!remote_client && exchandler.exthrowed) {
            send_close(socket, exchandler.exmsg);
            return 1;
        }
        goto reinit;
    }

    SWIGVM_params->dbname = (char*) g_database_name.c_str();
    SWIGVM_params->dbversion = (char*) g_database_version.c_str();
    SWIGVM_params->script_name = (char*) g_script_name.c_str();
    SWIGVM_params->script_schema = (char*) g_script_schema.c_str();
    SWIGVM_params->current_user = (char*) g_current_user.c_str();
    SWIGVM_params->current_schema = (char*) g_current_schema.c_str();
    SWIGVM_params->script_code = (char*) g_source_code.c_str();    
    SWIGVM_params->session_id = g_session_id;
    SWIGVM_params->statement_id = g_statement_id;
    SWIGVM_params->node_count = g_node_count;
    SWIGVM_params->node_id = g_node_id;
    SWIGVM_params->vm_id = g_vm_id;
    SWIGVM_params->singleCallMode = g_singleCallMode;

    SWIGVM *vm = NULL;
    try {
#ifdef PROTEGRITY_PLUGIN_CLIENT
        vm = new Protegrity(false);
#else
        if (strcmp(argv[2], "lang=python")==0)
        {
#ifdef ENABLE_PYTHON_VM
                vm = new PythonVM(false);
#else
                send_close(socket, "Unknown or unsupported VM type");
                return 1;
#endif
        } else if (strcmp(argv[2], "lang=r")==0)
        {
#ifdef ENABLE_R_VM
                vm = new RVM(false);
#else
            send_close(socket, "Unknown or unsupported VM type");
            return 1;
#endif
        } else if (strcmp(argv[2], "lang=java")==0)
        {
#ifdef ENABLE_JAVA_VM
            vm = new JavaVMach(false);
#else
            send_close(socket, "Unknown or unsupported VM type");
            return 1;
#endif
        } else if (strcmp(argv[2], "lang=cpp")==0)
        {
#ifdef ENABLE_CPP_VM
            vm = new CPPVM(false);
#else
            send_close(socket, "Unknown or unsupported VM type: CPP");
            return 1;
#endif
        } else {
            send_close(socket, "Unknown or unsupported VM type");
            return 1;
        }
#endif

        if (g_singleCallMode) {
            ExecutionGraph::EmptyDTO noArg; // used as dummy arg
            for (;;) {
                // in single call mode, after MT_RUN from the client,
                // EXASolution responds with a CALL message that specifies
                // the single call function to be made
                if (!send_run(socket)) {break;}
                assert(g_singleCallFunction != SC_FN_NIL);
                try {
                    std::string result;
                    switch (g_singleCallFunction)
                    {
                    case SC_FN_NIL:
                        break;
                    case SC_FN_DEFAULT_OUTPUT_COLUMNS:
                        result = vm->singleCall(g_singleCallFunction,noArg);
                        break;
                    case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
                        assert(!g_singleCall_ImportSpecificationArg.isEmpty());
                        result = vm->singleCall(g_singleCallFunction,g_singleCall_ImportSpecificationArg);
                        g_singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification();  // delete the last argument
                        break;
                    case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                        assert(!g_singleCall_StringArg.isEmpty());
                        result = vm->singleCall(g_singleCallFunction,g_singleCall_StringArg);
                        break;
                    }
                    send_return(socket,result);
                    if (!send_done(socket)) {
                        break;
                    }
                } catch (const swig_undefined_single_call_exception& ex) {
                   send_undefined_call(socket,ex.fn());
                }
            }
        } else {
            for(;;) {
                if (!send_run(socket))
                {
                    break;
                }
                SWIGVM_params->inp_force_finish = false;
                while(!vm->run()) {
                }
                if (!send_done(socket))
                {
                    break;
                }
            }
        }
        if (vm)
        {
            vm->shutdown();
            delete vm;
            vm = NULL;
        }
        send_finished(socket);
        std::cerr << "stm652: sent finish" << std::endl;
    } catch (std::exception &err) {
        send_close(socket, err.what()); socket.close();
#ifdef SWIGVM_LOG_CLIENT
        cerr << "### SWIGVM crashing with name '" << socket_name
             << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what() << endl;
#endif
        goto error;
    } catch (...) {
        send_close(socket, "Internal/Unknown error throwed"); socket.close();
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
    keep_checking = false;
    socket.close();
    if (!remote_client) {
        ::pthread_cancel(check_thread);
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
    if (!remote_client) {
        ::pthread_cancel(check_thread);
        ::unlink(socket_name_file);
    } else {
        ::sleep(3); // give other components time to shutdown
    }
    return 1;
}
