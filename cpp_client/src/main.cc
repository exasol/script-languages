#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include "wrapper.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include <fcntl.h>
#include <fstream>

#include <dlfcn.h>
#include <linux/limits.h>
#include <memory>
#include <stdexcept>
#include <openssl/md5.h>



using namespace UDFClient;
using namespace std;



static string socket_name;
static char *socket_name_str;
static string output_buffer;
static UDFClientExceptionHolder exchandler;
static pid_t my_pid;
static exascript_request request;
static exascript_response response;

static single_call_function_id g_singleCallFunction;
static UDFClient::ImportSpecification g_singleCall_ImportSpecificationArg;
static UDFClient::StringDTO g_singleCall_StringArg;



static bool remote_client;


static void external_process_check()
{
    if (remote_client) return;
    if (::access(&(socket_name_str[6]), F_OK) != 0) {
        ::sleep(1); // give me a chance to die with my parent process
        cerr << "exaudfclient aborting ... cannot access socket file " << socket_name_str+6 << "." << endl;
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

void UDFClient::socket_send(zmq::socket_t &socket, zmq::message_t &zmsg)
{
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

bool UDFClient::socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error)
{
    for (;;) {
        try {
            if (socket.recv(&zmsg) == true) {
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

static bool send_init(zmq::socket_t &socket, const string client_name, Metadata &meta)
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

    meta.connection_id = response.connection_id();
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

    meta.dbname = rep.database_name();
    meta.dbversion = rep.database_version();
    meta.script_name = rep.script_name();
    meta.script_schema = rep.script_schema();
    meta.script_code = rep.source_code();
    meta.session_id = rep.session_id();
    meta.statement_id = rep.statement_id();
    meta.node_count = rep.node_count();
    meta.node_id = rep.node_id();
    meta.vm_id = rep.vm_id();
    //vm_type = rep.vm_type();


    meta.maximal_memory_limit = rep.maximal_memory_limit();
    struct rlimit d;
    d.rlim_cur = d.rlim_max = rep.maximal_memory_limit();
    if (setrlimit(RLIMIT_RSS, &d) != 0)
        cerr << "WARNING: Failed to set memory limit" << endl;
    d.rlim_cur = d.rlim_max = 0;    // 0 for no core dumps, RLIM_INFINITY to enable coredumps of any size
    if (setrlimit(RLIMIT_CORE, &d) != 0)
        cerr << "WARNING: Failed to set core limit" << endl;
    getrlimit(RLIMIT_NOFILE,&d);
    if (d.rlim_max < 32768)
    {
        cerr << "WARNING: Reducing RLIMIT_NOFILE below 32768" << endl;
    }
    d.rlim_cur = d.rlim_max = std::min(32768,(int)d.rlim_max);
    if (setrlimit(RLIMIT_NOFILE, &d) != 0)
        cerr << "WARNING: Failed to set nofile limit" << endl;
    d.rlim_cur = d.rlim_max = 32768;
    if (setrlimit(RLIMIT_NPROC, &d) != 0)
    {
        cerr << "WARNING: Failed to set nproc limit to 32k trying 8k ..." << endl;
        d.rlim_cur = d.rlim_max = 8192;
        if (setrlimit(RLIMIT_NPROC, &d) != 0)
        cerr << "WARNING: Failed to set nproc limit" << endl;
    }

    { /* send meta request */
        request.Clear();
        request.set_type(MT_META);
        request.set_connection_id(meta.connection_id);
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
        meta.singleCallMode = rep.single_call_mode();
        meta.inp_iter_type = (IteratorType)(rep.input_iter_type());
        meta.out_iter_type = (IteratorType)(rep.output_iter_type());
        for (int col = 0; col < rep.input_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.input_columns(col);
            meta.inp_names.push_back(coldef.name());
            meta.inp_types.push_back(ColumnType());
            ColumnType &coltype = meta.inp_types.back();
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
            meta.out_names.push_back(coldef.name());
            meta.out_types.push_back(ColumnType());
            ColumnType &coltype = meta.out_types.back();
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

static void send_close(zmq::socket_t &socket, const string &exmsg, const Metadata& meta)
{
    request.Clear();
    request.set_type(MT_CLOSE);
    request.set_connection_id(meta.connection_id);
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
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        else if (response.type() != MT_FINISHED)
            throw LanguagePlugin::exception("Wrong response type, should be finished");
    }
}

static bool send_run(zmq::socket_t &socket, const Metadata& meta)
{
    {
        /* send done request */
        request.Clear();
        request.set_type(MT_RUN);
        request.set_connection_id(meta.connection_id);
        if (!request.SerializeToString(&output_buffer))
        {
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        }
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw LanguagePlugin::exception(response.close().exception_message().c_str());
            throw LanguagePlugin::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (meta.singleCallMode && response.type() == MT_CALL) {
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
                    throw LanguagePlugin::exception("internal error: SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC without import specification");
                }
                const import_specification_rep& is_proto = sc.import_specification();
                g_singleCall_ImportSpecificationArg = UDFClient::ImportSpecification(is_proto.is_subselect());
                if (is_proto.has_connection_information())
                {
                    const connection_information_rep& ci_proto = is_proto.connection_information();
                    UDFClient::ConnectionInformation connection_info(ci_proto.kind(), ci_proto.address(), ci_proto.user(), ci_proto.password());
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
                    throw LanguagePlugin::exception("internal error: SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL without json arg");
                }
                const std::string json = sc.json_arg();
                g_singleCall_StringArg = UDFClient::StringDTO(json);
                break;
            }

            return true;
        } else if (response.type() != MT_RUN) {
            throw LanguagePlugin::exception("Wrong response type, should be done");
        }
    }
    return true;
}


static bool send_return(zmq::socket_t &socket, std::string& result, const Metadata& meta)
{
    {   /* send return request */
        request.Clear();
        request.set_type(MT_RETURN);
        ::exascript_return_req* rr = new ::exascript_return_req();
        rr->set_result(result.c_str());
        request.set_allocated_call_result(rr);
        request.set_connection_id(meta.connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw LanguagePlugin::exception(response.close().exception_message().c_str());
            throw LanguagePlugin::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (response.type() != MT_RETURN) {
            throw LanguagePlugin::exception("Wrong response type, should be MT_RETURN");
        }
    }
    return true;
}

static void send_undefined_call(zmq::socket_t &socket, const std::string& fn, const Metadata& meta)
{
    {   /* send return request */
        request.Clear();
        request.set_type(MT_UNDEFINED_CALL);
        ::exascript_undefined_call_req* uc = new ::exascript_undefined_call_req();
        uc->set_remote_fn(fn);
        request.set_allocated_undefined_call(uc);
        request.set_connection_id(meta.connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        if (response.type() != MT_UNDEFINED_CALL) {
            throw LanguagePlugin::exception("Wrong response type, should be MT_UNDEFINED_CALL");
        }
    }
}


static bool send_done(zmq::socket_t &socket, const Metadata& meta)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_DONE);
        request.set_connection_id(meta.connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if (!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw LanguagePlugin::exception(response.close().exception_message().c_str());
            throw LanguagePlugin::exception("Wrong response type, got empty close response");
        } else if (response.type() == MT_CLEANUP) {
            return false;
        } else if (response.type() != MT_DONE)
            throw LanguagePlugin::exception("Wrong response type, should be done");
    }
    return true;
}

static void send_finished(zmq::socket_t &socket, const Metadata &meta)
{
    {   /* send done request */
        request.Clear();
        request.set_type(MT_FINISHED);
        request.set_connection_id(meta.connection_id);
        if (!request.SerializeToString(&output_buffer))
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        socket_recv(socket, zmsg);
        response.Clear();
        if(!response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        if (response.type() == MT_CLOSE) {
            if (response.close().has_exception_message())
                throw LanguagePlugin::exception(response.close().exception_message().c_str());
            throw LanguagePlugin::exception("Wrong response type, got empty close response");
        } else if (response.type() != MT_FINISHED)
            throw LanguagePlugin::exception("Wrong response type, should be finished");
    }
}



std::string getExecutablePath()
{
    char buf[PATH_MAX+1];
    ssize_t count = readlink("/proc/self/exe", buf, PATH_MAX);
    if (count>0)
    {
        buf[count] = '\0';
        return string(buf);
    }
    abort();
}


class CPPPlugin : public LanguagePlugin {
public:

    bool mexec(const string& cmd_, string& result) {
        char buffer[128];
        stringstream cmd;
        cmd << "ulimit -v 500000; ";
        cmd << cmd_ << " 2>&1";

        FILE* pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) {
            result = "Cannot start command `" + cmd.str() + "`";
            return false;
        }
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL) {
                result += buffer;
            }
        }
        int s = pclose(pipe);
        if (s == -1)
        {
            return false;
        }
        if (WEXITSTATUS(s))
        {
            return false;
        }
        return true;
    }


    typedef void (*RUN_FUNC)(UDFClient::Metadata*, UDFClient::InputTable*, UDFClient::OutputTable*);
    typedef string (*DEFAULT_OUTPUT_COLUMNS_FUNC)(UDFClient::Metadata*);
    typedef string (*ADAPTER_CALL_FUNC)(Metadata* meta, const string input);
    typedef string (*IMPORT_ALIAS_FUNC)(Metadata* meta, const ImportSpecification& importSpecification);

    struct exception: std::exception {
        exception(const char *reason): m_reason(reason) { }
        virtual ~exception() throw() { }
        const char* what() const throw() { return m_reason.c_str(); }
    private:
        std::string m_reason;
    };

    set< vector<unsigned char> > m_importedScriptChecksums;

    Metadata& meta;
    InputTable iter;
    OutputTable res;
    void* handle;

    string getOptionLine(string& scriptCode, const string option, const string whitespace, const string lineEnd, size_t& pos) {
        string result;
        size_t startPos = scriptCode.find(option);
        if (startPos != string::npos) {
            size_t firstPos = startPos + option.length();
            firstPos = scriptCode.find_first_not_of(whitespace, firstPos);
            if (firstPos == string::npos) {
                stringstream ss;
                ss << "No values found for " << option << " statement";
                throw LanguagePlugin::exception(ss.str().c_str());
            }
            size_t lastPos = scriptCode.find_first_of(lineEnd + "\r\n", firstPos);
            if (lastPos == string::npos || scriptCode.compare(lastPos, lineEnd.length(), lineEnd) != 0) {
                stringstream ss;
                ss << "End of " << option << " statement not found";
                throw LanguagePlugin::exception(ss.str().c_str());
            }
            if (firstPos >= lastPos) {
                stringstream ss;
                ss << "No values found for " << option << " statement";
                throw LanguagePlugin::exception(ss.str().c_str());
            }
            size_t optionsEnd = scriptCode.find_last_not_of(whitespace, lastPos - 1);
            if (optionsEnd == string::npos || optionsEnd < firstPos) {
                stringstream ss;
                ss << "No values found for " << option << " statement";
                throw LanguagePlugin::exception(ss.str().c_str());
            }
            result = scriptCode.substr(firstPos, optionsEnd - firstPos + 1);
            scriptCode.erase(startPos, lastPos - startPos + 1);
        }
        pos = startPos;
        return result;
    }

    vector<unsigned char> scriptToMd5(const char *script) {
        MD5_CTX ctx;
        unsigned char md5[MD5_DIGEST_LENGTH];
        MD5_Init(&ctx);
        MD5_Update(&ctx, script, strlen(script));
        MD5_Final(md5, &ctx);
        return vector<unsigned char>(md5, md5 + sizeof(md5));
    }

    void importScripts() {

        const string whitespace = " \t\f\v";
        const string lineEnd = ";";
        size_t pos;
        // Attention: We must hash the parent script before modifying it (adding the
        // package definition). Otherwise we don't recognize if the script imports itself
        m_importedScriptChecksums.insert(scriptToMd5(meta.script_code.c_str()));
        while (true) {
            string scriptName = getOptionLine(meta.script_code, "%import", whitespace, lineEnd, pos);
            if (scriptName == "")
                break;

            const char *scriptCode = meta.moduleContent(scriptName.c_str());
            const char *exception = meta.checkException();
            if (exception)
                throw LanguagePlugin::exception(exception);
            if (m_importedScriptChecksums.insert(scriptToMd5(scriptCode)).second) {
                // Script has not been imported yet
                // If this imported script contains %import statements
                // they will be resolved in this while loop.
                meta.script_code.insert(pos, scriptCode);
            }
        }
    }


    CPPPlugin(Metadata&meta_)
        : meta(meta_),
          iter(meta),
          res(meta, &iter)
    {
        string myPath = getExecutablePath();
        string myFolder = myPath.substr(0,myPath.find_last_of('/'));
        {
            stringstream cmd;
            cmd << "cp  " << myFolder << "/*.h /tmp/";

            if (::system(cmd.str().c_str()))
            {
                cerr << "Some error when copying header file" << endl;
                cerr << "current dir: " << endl;
                if (system("pwd")) {}
                abort();
            }
        }
        importScripts();
        const string whitespace = " \t\f\v";
        const string lineEnd = ";";
        size_t nextOptionPos = 0;


        string LDFLAGS = getOptionLine(meta.script_code,"%compilerflags",whitespace,lineEnd,nextOptionPos);

        std::ofstream out("/tmp/code.cc");
        out << "#include \"wrapper.h\"" << std::endl;

        out << meta.script_code << std::endl;
        out.close();

        {
            stringstream cmd;
            cmd << "g++ -shared -fPIC -o /tmp/libcode.so /tmp/code.cc";
            cmd << " -I" << myFolder;
            cmd << " " << LDFLAGS;

            string msg;
            if (!mexec(cmd.str(), msg))
            {

                throw LanguagePlugin::exception(("Error when compiling script code:\n"+cmd.str()+"\n\n"+msg).c_str());
            }
        }

#if 1
        {
           if (::system("nm /tmp/libcode.so")) {}
        }
#endif
        handle = dlopen("/tmp/libcode.so",RTLD_NOW);

        if (handle == NULL)
        {
            throw LanguagePlugin::exception( dlerror() );
        };


    }

protected:

    virtual ~CPPPlugin() { }
public:
    virtual void destroy() {delete this;}
    virtual bool run()
    {
        if (meta.singleCallMode)
        {
            throw LanguagePlugin::exception("calling RUN in single call mode");
        }
        RUN_FUNC run_cpp;
        char *error;
        run_cpp = (RUN_FUNC)dlsym(handle, "_Z7run_cppRKN9UDFClient8MetadataERNS_10InputTableERNS_11OutputTableE");
        if ((error = dlerror()) != NULL)  {
            stringstream sb;
            sb << "Error when trying to load function \"run_cpp\": " << endl << error;
            throw LanguagePlugin::exception(sb.str().c_str());
        }
        (*run_cpp)(&meta,&iter,&res);
        res.next(); // in case next() was not called in the UDF, the database will wait forever (or a timeout occurs)
        res.flush();
        return true;
    }
    virtual std::string singleCall(single_call_function_id fn, const UDFClient::ScriptDTO& args)
    {
        DEFAULT_OUTPUT_COLUMNS_FUNC defaultOutputColumnsFunc = NULL;
        ADAPTER_CALL_FUNC adapterCallFunc = NULL;
        IMPORT_ALIAS_FUNC importAliasFunc = NULL;
        UDFClient::StringDTO* stringDTO = NULL;
        UDFClient::ImportSpecification* importDTO = NULL;
        char *error = NULL;

        switch (fn)
        {
        case SC_FN_DEFAULT_OUTPUT_COLUMNS:
            defaultOutputColumnsFunc = (DEFAULT_OUTPUT_COLUMNS_FUNC)dlsym(handle, "_Z23getDefaultOutputColumnsB5cxx11RKN9UDFClient8MetadataE");
            if ((error = dlerror()) != NULL)
            {
                stringstream sb;
                sb << "Error when trying to load singleCall function: " << endl << error;
                throw LanguagePlugin::exception(sb.str().c_str());
            }
            return (*defaultOutputColumnsFunc)(&meta);
            break;
        case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
            adapterCallFunc = (ADAPTER_CALL_FUNC)dlsym(handle, "_Z11adapterCallRKN9UDFClient8MetadataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
            if ((error = dlerror()) != NULL)
            {
                stringstream sb;
                sb << "Error when trying to load singleCall function: " << endl << error;
                throw LanguagePlugin::exception(sb.str().c_str());
            }
            stringDTO = (UDFClient::StringDTO*)&args;
            assert(stringDTO != NULL);
            return (*adapterCallFunc)(&meta,stringDTO->getArg());
            break;
        case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
            importAliasFunc = (IMPORT_ALIAS_FUNC)dlsym(handle,"_Z24generateSqlForImportSpecB5cxx11RKN9UDFClient8MetadataERKNS_19ImportSpecificationE");
            if ((error = dlerror()) != NULL)
            {
                stringstream sb;
                sb << "Error when trying to load singleCall function: " << endl << error;
                throw LanguagePlugin::exception(sb.str().c_str());
            }
            importDTO = (UDFClient::ImportSpecification*)&args;
            assert(importDTO != NULL);
            return (*importAliasFunc)(&meta,*importDTO);
            break;
        default:
        {
            stringstream sb;
            sb << "Unsupported singleCall function id: " << fn;
            throw LanguagePlugin::exception(sb.str().c_str());
        }
        }

        return "dummy";
    }
};





int main(int argc, char **argv) {

    if (::setenv("HOME", "/tmp", 1) == -1)
    {
        throw LanguagePlugin::exception("Failed to set HOME directory");
    }
    ::setlocale(LC_ALL, "en_US.utf8");

    socket_name = argv[1];
    socket_name_str = argv[1];
    char *socket_name_file = argv[1];
    remote_client = false;
    my_pid = ::getpid();

    Metadata meta;

    zmq::context_t context(1);

#ifdef LOG_CLIENT_ARGS
    for (int i = 0; i<argc; i++)
    {
        cerr << "zmqcontainerclient argv[" << i << "] = " << argv[i] << endl;
    }
#endif


    if (strncmp(socket_name_str, "tcp:", 4) == 0) {
            remote_client = true;
    }

    if (socket_name.length() > 6 && strncmp(socket_name_str, "ipc:", 4) == 0)
    {
        socket_name_file = &(socket_name_file[6]);
    }


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

    meta.sock = &socket;
    meta.exch = &exchandler;

    if (!send_init(socket, socket_name, meta)) {
        if (!remote_client && exchandler.has_exception) {
            send_close(socket, exchandler.exception_message, meta);
            return 1;
        }
        goto reinit;
    }

    LanguagePlugin *vm = NULL;
    try {
        vm = new CPPPlugin(meta);
        if (meta.singleCallMode) {
            UDFClient::EmptyDTO noArg; // used as dummy arg
            for (;;) {
                // in single call mode, after MT_RUN from the client,
                // EXASolution responds with a CALL message that specifies
                // the single call function to be made
                if (!send_run(socket, meta)) {break;}
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
                        g_singleCall_ImportSpecificationArg = UDFClient::ImportSpecification();  // delete the last argument
                        break;
                    case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                        assert(!g_singleCall_StringArg.isEmpty());
                        result = vm->singleCall(g_singleCallFunction,g_singleCall_StringArg);
                        break;
                    }
                    send_return(socket,result, meta);
                    if (!send_done(socket, meta)) {
                        break;
                    }
                } catch (const swig_undefined_single_call_exception& ex) {
                   send_undefined_call(socket,ex.fn(), meta);
                }
            }
        } else {
            for(;;) {
                if (!send_run(socket, meta))
                    break;
                meta.inp_force_finish = false;
                while(!vm->run());
                if (!send_done(socket, meta))
                    break;
            }
        }
        vm->destroy(); vm = NULL;
        send_finished(socket, meta);
    }
    catch (std::exception &err) {
        send_close(socket, err.what(), meta);
        goto error;
    }
    catch (...) {
        send_close(socket, "Internal/Unknown error throwed", meta);
        socket.close();
        goto error;
    }
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
        vm->destroy();
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
