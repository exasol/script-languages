#include "exaudflib/socket_high_level.h"

#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/zmqcontainer.pb.h"
#include "exaudflib/impl/msg_conversion.h"
#include "exaudflib/impl/global.h"
#include "exaudflib/swig/swig_common.h"
#include "exaudflib/vm/swig_vm.h"
#include <sys/resource.h>
#include "debug_message.h"

namespace exaudflib {
    namespace socket_high_level {
        //static exascript_vmtype vm_type;
        static exascript_request request;
        static exascript_response response;
        static std::string output_buffer;
    }
}


bool exaudflib::socket_high_level::send_init(zmq::socket_t &socket, const std::string client_name)
{
    exaudflib::socket_high_level::request.Clear();
    exaudflib::socket_high_level::request.set_type(MT_CLIENT);
    exaudflib::socket_high_level::request.set_connection_id(0);
    exascript_client *req = exaudflib::socket_high_level::request.mutable_client();
    req->set_client_name(client_name);
    if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer)) {
        exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1002: Communication error: failed to serialize data");
        return false;
    }
    zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                        exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
    exaudflib::socket_low_level::socket_send(socket, zmsg);

    zmq::message_t zmsgrecv;
    exaudflib::socket_high_level::response.Clear();
    if (!exaudflib::socket_low_level::socket_recv(socket, zmsgrecv, true))
        return false;
    if (!exaudflib::socket_high_level::response.ParseFromArray(zmsgrecv.data(), zmsgrecv.size())) {
        exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1003: Failed to parse data");
        return false;
    }

    exaudflib::global.SWIGVM_params_ref->connection_id = exaudflib::socket_high_level::response.connection_id();
    DBG_STREAM_MSG(std::cerr,"### SWIGVM connected with id " << 
        std::hex << exaudflib::global.SWIGVM_params_ref->connection_id);
    if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
        if (exaudflib::socket_high_level::response.close().has_exception_message())
            exaudflib::global.exchandler.setException(
                exaudflib::socket_high_level::response.close().exception_message().c_str());
        else exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1004: Connection closed by server");
        return false;
    }
    if (exaudflib::socket_high_level::response.type() != MT_INFO) {
        exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1005: Wrong message type, should be MT_INFO got "+
        exaudflib::msg_conversion::convert_message_type_to_string(exaudflib::socket_high_level::response.type()));
        return false;
    }
    const exascript_info &rep = exaudflib::socket_high_level::response.info();
    exaudflib::global.writeScriptParams(rep);
    //vm_type = rep.vm_type();


    exaudflib::global.SWIGVM_params_ref->maximal_memory_limit = rep.maximal_memory_limit();
    struct rlimit d;
    d.rlim_cur = d.rlim_max = rep.maximal_memory_limit();
    if (setrlimit(RLIMIT_RSS, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        std::cerr << "W-UDF-CL-LIB-1006: Failed to set memory limit" << std::endl;
#else
    throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1007: Failed to set memory limit");
#endif
    d.rlim_cur = d.rlim_max = 0;    // 0 for no core dumps, RLIM_INFINITY to enable coredumps of any size
    if (setrlimit(RLIMIT_CORE, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        std::cerr << "W-UDF-CL-LIB-1008: Failed to set core dump size limit" << std::endl;
#else
    throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1009: Failed to set core dump size limit");
#endif
    /* d.rlim_cur = d.rlim_max = 65536; */
    getrlimit(RLIMIT_NOFILE,&d);
    if (d.rlim_max < 32768)
    {
        //#ifdef SWIGVM_LOG_CLIENT
        std::cerr << "W-UDF-CL-LIB-1010: Reducing RLIMIT_NOFILE below 32768" << std::endl;
        //#endif
    }
    d.rlim_cur = d.rlim_max = std::min(32768,(int)d.rlim_max);
    if (setrlimit(RLIMIT_NOFILE, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
        std::cerr << "W-UDF-CL-LIB-1011: Failed to set nofile limit" << std::endl;
#else
    throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1012: Failed to set nofile limit");
#endif
    d.rlim_cur = d.rlim_max = 32768;
    if (setrlimit(RLIMIT_NPROC, &d) != 0)
    {
        std::cerr << "W-UDF-CL-LIB-1013: Failed to set nproc limit to 32k trying 8k ..." << std::endl;
        d.rlim_cur = d.rlim_max = 8192;
        if (setrlimit(RLIMIT_NPROC, &d) != 0)
#ifdef SWIGVM_LOG_CLIENT
            std::cerr << "W-UDF-CL-LIB-1014: Failed to set nproc limit" << std::endl;
#else
        throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1015: Failed to set nproc limit");
#endif
    }

    { /* send meta request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_META);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer)) {
            exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1016: Communication error: failed to serialize data");
            return false;
        }
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                             exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    } /* receive meta response */
    {   zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if (!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size())) {
            exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1017: Communication error: failed to parse data");
            return false;
        }
        if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
            if (exaudflib::socket_high_level::response.close().has_exception_message())
                exaudflib::global.exchandler.setException(
                    exaudflib::socket_high_level::response.close().exception_message().c_str());
            else exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1018: Connection closed by server");
            return false;
        }
        if (exaudflib::socket_high_level::response.type() != MT_META) {
            exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1019: Wrong message type, should be META, got "+
            exaudflib::msg_conversion::convert_message_type_to_string(exaudflib::socket_high_level::response.type()));
            return false;
        }
        const exascript_metadata &rep = exaudflib::socket_high_level::response.meta();
        exaudflib::global.singleCallMode = rep.single_call_mode();
        exaudflib::global.SWIGVM_params_ref->inp_iter_type = (SWIGVMContainers::SWIGVM_itertype_e)(rep.input_iter_type());
        exaudflib::global.SWIGVM_params_ref->out_iter_type = (SWIGVMContainers::SWIGVM_itertype_e)(rep.output_iter_type());
        for (int col = 0; col < rep.input_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.input_columns(col);
            exaudflib::global.SWIGVM_params_ref->inp_names->push_back(coldef.name());
            exaudflib::global.SWIGVM_params_ref->inp_types->push_back(SWIGVMContainers::SWIGVM_columntype_t());
            SWIGVMContainers::SWIGVM_columntype_t &coltype = exaudflib::global.SWIGVM_params_ref->inp_types->back();
            coltype.len = 0; coltype.prec = 0; coltype.scale = 0;
            coltype.type_name = coldef.type_name();
            switch (coldef.type()) {
                case PB_UNSUPPORTED:
                    exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1020: Unsupported input column type found");
                    return false;
                case PB_DOUBLE:
                    coltype.type = SWIGVMContainers::DOUBLE;
                    break;
                case PB_INT32:
                    coltype.type = SWIGVMContainers::INT32;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_INT64:
                    coltype.type = SWIGVMContainers::INT64;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_NUMERIC:
                    coltype.type = SWIGVMContainers::NUMERIC;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_TIMESTAMP:
                    coltype.type = SWIGVMContainers::TIMESTAMP;
                    break;
                case PB_DATE:
                    coltype.type = SWIGVMContainers::DATE;
                    break;
                case PB_STRING:
                    coltype.type = SWIGVMContainers::STRING;
                    coltype.len = coldef.size();
                    break;
                case PB_BOOLEAN:
                    coltype.type = SWIGVMContainers::BOOLEAN;
                    break;
                default:
                    exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1021: Unknown input column type found, got "+coldef.type());
                    return false;
            }
        }
        for (int col = 0; col < rep.output_columns_size(); ++col) {
            const exascript_metadata_column_definition &coldef = rep.output_columns(col);
            exaudflib::global.SWIGVM_params_ref->out_names->push_back(coldef.name());
            exaudflib::global.SWIGVM_params_ref->out_types->push_back(SWIGVMContainers::SWIGVM_columntype_t());
            SWIGVMContainers::SWIGVM_columntype_t &coltype = exaudflib::global.SWIGVM_params_ref->out_types->back();
            coltype.len = 0; coltype.prec = 0; coltype.scale = 0;
            coltype.type_name = coldef.type_name();
            switch (coldef.type()) {
                case PB_UNSUPPORTED:
                    exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1022: Unsupported output column type found");
                    return false;
                case PB_DOUBLE:
                    coltype.type = SWIGVMContainers::DOUBLE;
                    break;
                case PB_INT32:
                    coltype.type = SWIGVMContainers::INT32;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_INT64:
                    coltype.type = SWIGVMContainers::INT64;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_NUMERIC:
                    coltype.type = SWIGVMContainers::NUMERIC;
                    coltype.prec = coldef.precision();
                    coltype.scale = coldef.scale();
                    break;
                case PB_TIMESTAMP:
                    coltype.type = SWIGVMContainers::TIMESTAMP;
                    break;
                case PB_DATE:
                    coltype.type = SWIGVMContainers::DATE;
                    break;
                case PB_STRING:
                    coltype.type = SWIGVMContainers::STRING;
                    coltype.len = coldef.size();
                    break;
                case PB_BOOLEAN:
                    coltype.type = SWIGVMContainers::BOOLEAN;
                    break;
                default:
                    exaudflib::global.exchandler.setException("F-UDF-CL-LIB-1023: Unknown output column type found, got "+coldef.type());
                    return false;
            }
        }
    }
    return true;
}

void exaudflib::socket_high_level::send_close(zmq::socket_t &socket, const std::string &exmsg)
{
    exaudflib::socket_high_level::request.Clear();
    exaudflib::socket_high_level::request.set_type(MT_CLOSE);
    exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
    exascript_close *req = 
        exaudflib::socket_high_level::request.mutable_close();
    if (exmsg != "") req->set_exception_message(exmsg);
    exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer);
    zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                        exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
    exaudflib::socket_low_level::socket_send(socket, zmsg);

    { /* receive finished response, so we know that the DB knows that we are going to close and
         all potential exceptions have been received on DB side */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if(!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size())) {
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1024: Communication error: failed to parse data");
        }
        else if (exaudflib::socket_high_level::response.type() != MT_FINISHED) {
            throw 
                SWIGVMContainers::SWIGVM::exception(
                    "F-UDF-CL-LIB-1025: Wrong response type, should be MT_FINISHED, got "+
                    exaudflib::msg_conversion::convert_message_type_to_string(
                        exaudflib::socket_high_level::response.type()));
        }
    }
}

bool exaudflib::socket_high_level::send_run(zmq::socket_t &socket)
{
    {
        /* send done request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_RUN);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer))
        {
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1026: Communication error: failed to serialize data");
        }
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                            exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if (!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1027: Communication error: failed to parse data");
        if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
            if (exaudflib::socket_high_level::response.close().has_exception_message())
                throw SWIGVMContainers::SWIGVM::exception(
                    exaudflib::socket_high_level::response.close().exception_message().c_str());
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1028: Wrong response type, got empty MT_CLOSE");
        } else if (exaudflib::socket_high_level::response.type() == MT_CLEANUP) {
            return false;
        } else if (exaudflib::global.singleCallMode && exaudflib::socket_high_level::response.type() == MT_CALL) {
            assert(exaudflib::global.singleCallMode);
            exascript_single_call_rep sc = exaudflib::socket_high_level::response.call();
            exaudflib::global.singleCallFunction = static_cast<SWIGVMContainers::single_call_function_id_e>(sc.fn());

            switch (exaudflib::global.singleCallFunction)
            {
                case SWIGVMContainers::single_call_function_id_e::SC_FN_NIL:
                case SWIGVMContainers::single_call_function_id_e::SC_FN_DEFAULT_OUTPUT_COLUMNS:
                        break;
                case SWIGVMContainers::single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
                {

                    if (!sc.has_import_specification())
                    {
                        throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1029: internal error SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC without import specification");
                    }
                    const import_specification_rep& is_proto = sc.import_specification();
                    exaudflib::global.singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification(is_proto.is_subselect());
                    if (is_proto.has_connection_information())
                    {
                        const connection_information_rep& ci_proto = is_proto.connection_information();
                        ExecutionGraph::ConnectionInformation connection_info(ci_proto.kind(), ci_proto.address(), ci_proto.user(), ci_proto.password());
                        exaudflib::global.singleCall_ImportSpecificationArg.setConnectionInformation(connection_info);
                    }
                    if (is_proto.has_connection_name())
                    {
                        exaudflib::global.singleCall_ImportSpecificationArg.setConnectionName(is_proto.connection_name());
                    }
                    for (int i=0; i<is_proto.subselect_column_specification_size(); i++)
                    {
                        const ::exascript_metadata_column_definition& cdef = is_proto.subselect_column_specification(i);
                        const ::std::string& cname = cdef.name();
                        const ::std::string& ctype = cdef.type_name();
                        exaudflib::global.singleCall_ImportSpecificationArg.appendSubselectColumnName(cname);
                        exaudflib::global.singleCall_ImportSpecificationArg.appendSubselectColumnType(ctype);
                    }
                    for (int i=0; i<is_proto.parameters_size(); i++)
                    {
                        const ::key_value_pair& kvp = is_proto.parameters(i);
                        exaudflib::global.singleCall_ImportSpecificationArg.addParameter(kvp.key(), kvp.value());
                    }
                }
                break;
                case SWIGVMContainers::single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC:
                {
                    if (!sc.has_export_specification())
                    {
                        throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1030: internal error SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC without export specification");
                    }
                    const export_specification_rep& es_proto = sc.export_specification();
                    exaudflib::global.singleCall_ExportSpecificationArg = ExecutionGraph::ExportSpecification();
                    if (es_proto.has_connection_information())
                    {
                        const connection_information_rep& ci_proto = es_proto.connection_information();
                        ExecutionGraph::ConnectionInformation connection_info(ci_proto.kind(), ci_proto.address(), ci_proto.user(), ci_proto.password());
                        exaudflib::global.singleCall_ExportSpecificationArg.setConnectionInformation(connection_info);
                    }
                    if (es_proto.has_connection_name())
                    {
                        exaudflib::global.singleCall_ExportSpecificationArg.setConnectionName(es_proto.connection_name());
                    }
                    for (int i=0; i<es_proto.parameters_size(); i++)
                    {
                        const ::key_value_pair& kvp = es_proto.parameters(i);
                        exaudflib::global.singleCall_ExportSpecificationArg.addParameter(kvp.key(), kvp.value());
                    }
                    exaudflib::global.singleCall_ExportSpecificationArg.setTruncate(es_proto.has_truncate());
                    exaudflib::global.singleCall_ExportSpecificationArg.setReplace(es_proto.has_replace());
                    if (es_proto.has_created_by())
                    {
                        exaudflib::global.singleCall_ExportSpecificationArg.setCreatedBy(es_proto.created_by());
                    }
                    for (int i=0; i<es_proto.source_column_names_size(); i++)
                    {
                        const std::string name = es_proto.source_column_names(i);
                        exaudflib::global.singleCall_ExportSpecificationArg.addSourceColumnName(name);
                    }
                }
                break;
                case SWIGVMContainers::single_call_function_id_e::SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                    if (!sc.has_json_arg())
                    {
                        throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1031: internal error SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL without json arg");
                    }
                    const std::string json = sc.json_arg();
                    exaudflib::global.singleCall_StringArg = ExecutionGraph::StringDTO(json);
                    break;
            }

            return true;
        } else if (exaudflib::socket_high_level::response.type() != MT_RUN) {
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1032: Wrong response type, should be MT_RUN, got "+
            exaudflib::msg_conversion::convert_message_type_to_string(exaudflib::socket_high_level::response.type()));
        }
    }
    return true;
}

bool exaudflib::socket_high_level::send_return(zmq::socket_t &socket, const char* result)
{
    assert(result != nullptr);
    {   /* send return request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_RETURN);
        ::exascript_return_req* rr = new ::exascript_return_req();
        rr->set_result(result);
        exaudflib::socket_high_level::request.set_allocated_call_result(rr);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1033: Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                            exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if (!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1034: Communication error: failed to parse data");
        if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
            if (exaudflib::socket_high_level::response.close().has_exception_message())
                throw SWIGVMContainers::SWIGVM::exception(
                    exaudflib::socket_high_level::response.close().exception_message().c_str());
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1035: Wrong response type, got empty close response");
        } else if (exaudflib::socket_high_level::response.type() == MT_CLEANUP) {
            return false;
        } else if (exaudflib::socket_high_level::response.type() != MT_RETURN) {
            throw SWIGVMContainers::SWIGVM::exception(
                "F-UDF-CL-LIB-1036: Wrong response type, should be MT_RETURN, got "+
                    exaudflib::msg_conversion::convert_message_type_to_string(
                        exaudflib::socket_high_level::response.type()));
        }
    }
    return true;
}

void send_undefined_call(zmq::socket_t &socket, const std::string& fn)
{
    {   /* send return request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_UNDEFINED_CALL);
        ::exascript_undefined_call_req* uc = new ::exascript_undefined_call_req();
        uc->set_remote_fn(fn);
        exaudflib::socket_high_level::request.set_allocated_undefined_call(uc);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1037: Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                            exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    } { /* receive return response */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if (!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1038: Communication error: failed to parse data");
        if (exaudflib::socket_high_level::response.type() != MT_UNDEFINED_CALL) {
            throw SWIGVMContainers::SWIGVM::exception(
                "F-UDF-CL-LIB-1039: Wrong response type, should be MT_UNDEFINED_CALL, got "+
                    exaudflib::msg_conversion::convert_message_type_to_string(
                        exaudflib::socket_high_level::response.type()));
        }
    }
}

bool exaudflib::socket_high_level::send_done(zmq::socket_t &socket)
{
    {   /* send done request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_DONE);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1040: Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                            exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    }
    { /* receive done response */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if (!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1041: Communication error: failed to parse data");
        if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
            if (exaudflib::socket_high_level::response.close().has_exception_message())
                throw SWIGVMContainers::SWIGVM::exception(
                    exaudflib::socket_high_level::response.close().exception_message().c_str());
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1042: Wrong response type, got empty close response");
        } else if (exaudflib::socket_high_level::response.type() == MT_CLEANUP) {
            return false;
        } else if (exaudflib::socket_high_level::response.type() != MT_DONE)
            throw SWIGVMContainers::SWIGVM::exception(
                "F-UDF-CL-LIB-1043: Wrong response type, should be MT_DONE, got "+
                    exaudflib::msg_conversion::convert_message_type_to_string(
                        exaudflib::socket_high_level::response.type()));
    }
    return true;
}

void exaudflib::socket_high_level::send_finished(zmq::socket_t &socket)
{
    {   /* send done request */
        exaudflib::socket_high_level::request.Clear();
        exaudflib::socket_high_level::request.set_type(MT_FINISHED);
        exaudflib::socket_high_level::request.set_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id);
        if (!exaudflib::socket_high_level::request.SerializeToString(&exaudflib::socket_high_level::output_buffer))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1044: Communication error: failed to serialize data");
        zmq::message_t zmsg((void*)exaudflib::socket_high_level::output_buffer.c_str(),
                            exaudflib::socket_high_level::output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(socket, zmsg);
    } { /* receive done response */
        zmq::message_t zmsg;
        exaudflib::socket_low_level::socket_recv(socket, zmsg);
        exaudflib::socket_high_level::response.Clear();
        if(!exaudflib::socket_high_level::response.ParseFromArray(zmsg.data(), zmsg.size()))
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1045: Communication error: failed to parse data");
        if (exaudflib::socket_high_level::response.type() == MT_CLOSE) {
            if (exaudflib::socket_high_level::response.close().has_exception_message())
                throw SWIGVMContainers::SWIGVM::exception(exaudflib::socket_high_level::response.close().exception_message().c_str());
            throw SWIGVMContainers::SWIGVM::exception("F-UDF-CL-LIB-1046: Wrong response type, got empty MT_CLOSE");
        } else if (exaudflib::socket_high_level::response.type() != MT_FINISHED)
            throw SWIGVMContainers::SWIGVM::exception(
                "F-UDF-CL-LIB-1047: Wrong response type, should be MT_FINISHED, got"+
                exaudflib::msg_conversion::convert_message_type_to_string(
                    exaudflib::socket_high_level::response.type()));
    }
}
