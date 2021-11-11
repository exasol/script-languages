#ifndef EXAUDFCLIENT_SWIGMETADATA_H
#define EXAUDFCLIENT_SWIGMETADATA_H

#include "exaudflib/swig/swig_common.h"
#include "exaudflib/swig/swig_meta_data.h"
#include "exaudflib/zmqcontainer.pb.h"
#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/impl/msg_conversion.h"
#include "exaudflib/impl/global.h"
#include <string>
#include <vector>
#include <sstream>

namespace SWIGVMContainers {

class SWIGMetadata_Impl : public SWIGMetadata {
public:
    SWIGMetadata_Impl():
        SWIGMetadata(false),
        m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
        m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
        m_exch(exaudflib::global.SWIGVM_params_ref->exch),
        m_db_name(exaudflib::global.SWIGVM_params_ref->dbname),
        m_db_version(exaudflib::global.SWIGVM_params_ref->dbversion),
        m_script_name(exaudflib::global.SWIGVM_params_ref->script_name),
        m_script_schema(exaudflib::global.SWIGVM_params_ref->script_schema),
        m_current_user(exaudflib::global.SWIGVM_params_ref->current_user),
        m_current_schema(exaudflib::global.SWIGVM_params_ref->current_schema),
        m_scope_user(exaudflib::global.SWIGVM_params_ref->scope_user),
        m_script_code(exaudflib::global.SWIGVM_params_ref->script_code),
        m_session_id(exaudflib::global.SWIGVM_params_ref->session_id),
        m_statement_id(exaudflib::global.SWIGVM_params_ref->statement_id),
        m_node_count(exaudflib::global.SWIGVM_params_ref->node_count),
        m_node_id(exaudflib::global.SWIGVM_params_ref->node_id),
        m_vm_id(exaudflib::global.SWIGVM_params_ref->vm_id),
        m_input_names(*(exaudflib::global.SWIGVM_params_ref->inp_names)),
        m_input_types(*(exaudflib::global.SWIGVM_params_ref->inp_types)),
        m_input_iter_type(exaudflib::global.SWIGVM_params_ref->inp_iter_type),
        m_output_names(*(exaudflib::global.SWIGVM_params_ref->out_names)),
        m_output_types(*(exaudflib::global.SWIGVM_params_ref->out_types)),
        m_output_iter_type(exaudflib::global.SWIGVM_params_ref->out_iter_type),
        m_memory_limit(exaudflib::global.SWIGVM_params_ref->maximal_memory_limit),
        m_vm_type(exaudflib::global.SWIGVM_params_ref->vm_type),
        m_is_emitted(*(exaudflib::global.SWIGVM_params_ref->is_emitted)),
        m_pluginLanguageName(exaudflib::global.SWIGVM_params_ref->pluginName),
        m_pluginURI(exaudflib::global.SWIGVM_params_ref->pluginURI),
        m_outputAddress(exaudflib::global.SWIGVM_params_ref->outputAddress)
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
            m_exch->setException("F-UDF-CL-LIB-1048: Communication error: failed to serialize data");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        zmq::message_t zmsg_req((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        exaudflib::socket_low_level::socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("F-UDF-CL-LIB-1049: Communication error: failed to parse data");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("F-UDF-CL-LIB-1050: Internal error: wrong message type, got "+
            exaudflib::msg_conversion::convert_message_type_to_string(response.type()));
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            m_exch->setException(rep.exception_message().c_str());
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        if (!rep.has_connection_information()) {
            m_exch->setException("F-UDF-CL-LIB-1051: Internal error: No connection information returned");
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
            m_exch->setException("F-UDF-CL-LIB-1052: Communication error: failed to serialize data");
            return NULL;
        }
        zmq::message_t zmsg_req((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
        exaudflib::socket_low_level::socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        exaudflib::socket_low_level::socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("F-UDF-CL-LIB-1053: Communication error: failed to parse data");
            return NULL;
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("F-UDF-CL-LIB-1054: Internal error: wrong message type, should MT_IMPORT, got "+
            exaudflib::msg_conversion::convert_message_type_to_string(response.type()));
            return NULL;
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            m_exch->setException(rep.exception_message().c_str());
            return NULL;
        }
        if (!rep.has_source_code()) {
            m_exch->setException("F-UDF-CL-LIB-1055: Internal error: No source code returned");
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

} //namespace SWIGVMContainers

#endif //EXAUDFCLIENT_SWIGMETADATA_H
