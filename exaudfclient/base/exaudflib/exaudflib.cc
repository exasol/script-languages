#include <sys/types.h>
#include <sys/stat.h>

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

#include "debug_message.h"

// swig lib
#include <limits>
#include "exaudflib/zmqcontainer.pb.h"
#include "script_data_transfer_objects_wrapper.h"
#include <unistd.h>


#include "exaudflib/impl/exaudflib_check.h"
#include "exaudflib/impl/exaudflib_socket_low_level.h"
#include "exaudflib/impl/exaudflib_msg_conversion.h"
#include "exaudflib/impl/exaudflib_global.h"

#ifdef PROTEGRITY_PLUGIN_CLIENT
#include <protegrityclient.h>
#endif

using namespace SWIGVMContainers;
using namespace std;
using namespace google::protobuf;

#ifndef PROTEGRITY_PLUGIN_CLIENT
__thread SWIGVM_params_t* SWIGVMContainers::SWIGVM_params; // this is not used in the file, but defined to satisfy the "extern" requirement from exaudflib.h
#endif


static pid_t my_pid; //parent_pid,


#ifndef NDEBUG
#define SWIGVM_LOG_CLIENT
#endif
//#define SWIGVM_LOG_CLIENT
//#define LOG_COMMUNICATION


void print_args(int argc,char**argv){
    for (int i = 0; i<argc; i++)
    {
        cerr << "zmqcontainerclient argv[" << i << "] = " << argv[i] << endl;
    }
}


void delete_vm(SWIGVM*& vm){
    if (vm != nullptr)
    {
        delete vm;
        vm = nullptr;
    }
}

void stop_all(zmq::socket_t& socket){
    socket.close();
    exaudflib_check::stop_check_thread();
    if (!exaudflib_check::get_remote_client()) {
        exaudflib_check::cancel_check_thread();
        ::unlink(exaudflib_check::get_socket_name_file());
    } else {
        ::sleep(3); // give other components time to shutdown
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
        exaudflib_socket_low_level::socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        exaudflib_socket_low_level::socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("F-UDF-CL-LIB-1049: Communication error: failed to parse data");
            return new ExecutionGraph::ConnectionInformationWrapper(ExecutionGraph::ConnectionInformation());
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("F-UDF-CL-LIB-1050: Internal error: wrong message type, got "+
                msg_conversion::convert_message_type_to_string(response.type()));
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
        exaudflib_socket_low_level::socket_send(m_socket, zmsg_req);
        zmq::message_t zmsg_rep;
        exaudflib_socket_low_level::socket_recv(m_socket, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            m_exch->setException("F-UDF-CL-LIB-1053: Communication error: failed to parse data");
            return NULL;
        }
        if (response.type() != MT_IMPORT) {
            m_exch->setException("F-UDF-CL-LIB-1054: Internal error: wrong message type, should MT_IMPORT, got "+
                                    msg_conversion::convert_message_type_to_string(response.type()));
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



class SWIGGeneralIterator {
    protected:
        SWIGVMExceptionHandler *m_exch;
    public:
//        SWIGGeneralIterator(SWIGVMExceptionHandler *exch): m_exch(exch) { }
        SWIGGeneralIterator()
            : m_exch(exaudflib::global.SWIGVM_params_ref->exch)
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
            sb << "F-UDF-CL-LIB-1056: Internal error: not enough nulls in packet: wanted index " << (null_index + m_column_count - 1)
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
              case UNSUPPORTED: m_exch->setException("F-UDF-CL-LIB-1057: Unsupported data type found"); return;
              case DOUBLE: m_col_offsets[current_column] = m_values_per_row.doubles++; break;
              case INT32: m_col_offsets[current_column] = m_values_per_row.int32s++; break;
              case INT64: m_col_offsets[current_column] = m_values_per_row.int64s++; break;
              case NUMERIC:
              case TIMESTAMP:
              case DATE:
              case STRING: m_col_offsets[current_column] = m_values_per_row.strings++; break;
              case BOOLEAN: m_col_offsets[current_column] = m_values_per_row.bools++; break;
              default: m_exch->setException("F-UDF-CL-LIB-1058: Unknown data type found, got "+it->type); return;
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
                m_exch->setException("F-UDF-CL-LIB-1059: Communication error: failed to serialize data");
                return;
            }
            zmq::message_t zmsg((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
            exaudflib_socket_low_level::socket_send(m_socket, zmsg);
        } {
            zmq::message_t zmsg;
            exaudflib_socket_low_level::socket_recv(m_socket, zmsg);
            m_next_response.Clear();
            if (!m_next_response.ParseFromArray(zmsg.data(), zmsg.size())) {
                m_exch->setException("F-UDF-CL-LIB-1060: Communication error: failed to parse data");
                return;
            }
            if (m_next_response.connection_id() != m_connection_id) {
                std::stringstream sb;
                sb << "F-UDF-CL-LIB-1061: Communication error: wrong connection id, expected "
                   << m_connection_id << " got " << m_next_response.connection_id(); 
                m_exch->setException(sb.str());
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
                    } else m_exch->setException("F-UDF-CL-LIB-1062: Unknown error occured");
                } else {
                    m_exch->setException(rep.exception_message().c_str());
                }
                return;
            }
            if ((reset && (m_next_response.type() != MT_RESET)) ||
                    (!reset && (m_next_response.type() != MT_NEXT)))
            {
                m_exch->setException("F-UDF-CL-LIB-1063: Communication error: wrong message type, got "+
                                        msg_conversion::convert_message_type_to_string(m_next_response.type()));
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
        sb << "F-UDF-CL-LIB-1064: Internal error: not enough " << tp << otype << ts << " in packet: wanted index "
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
        m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
        m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
        m_column_count(exaudflib::global.SWIGVM_params_ref->inp_types->size()),
        m_col_offsets(exaudflib::global.SWIGVM_params_ref->inp_types->size()),
        m_current_row((uint64_t)-1),
        m_rows_completed(0),
        m_rows_group_completed(1),
        m_was_null(false),
        m_types(*(exaudflib::global.SWIGVM_params_ref->inp_types))
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
            m_exch->setException("E-UDF-CL-LIB-1065: Iteration finished");
            return false;
        }
        ++m_rows_completed;
        ++m_rows_group_completed;
        if (exaudflib::global.SWIGVM_params_ref->inp_force_finish)
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
        exaudflib::global.SWIGVM_params_ref->inp_force_finish = false;
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
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1066: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return 0.0; 
        }
        if (m_types[col].type != DOUBLE) {
            m_exch->setException("E-UDF-CL-LIB-1067: Wrong input column type, expected DOUBLE, got "+
                                msg_conversion::convert_type_to_string(m_types[col].type));
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
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1068: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return ""; 
        }
        if (m_types[col].type != STRING) {
            m_exch->setException("E-UDF-CL-LIB-1069: Wrong input column type, expected STRING, got "+
                                msg_conversion::convert_type_to_string(m_types[col].type));
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
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1070: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return 0; 
        }
        if (m_types[col].type != INT32) {
            m_exch->setException("E-UDF-CL-LIB-1071: Wrong input column type, expected INT32, got "+
                                msg_conversion::convert_type_to_string(m_types[col].type));
            m_was_null = true;
            return 0;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int32_size(), "int32");
        if (m_was_null) return 0;
        return m_next_response.next().table().data_int32(index);
    }
    inline int64_t getInt64(unsigned int col) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1072: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return 0LL; 
        }
        if (m_types[col].type != INT64) {
            m_exch->setException("E-UDF-CL-LIB-1073: Wrong input column type, expected INT64, got "+
                                msg_conversion::convert_type_to_string(m_types[col].type));
            m_was_null = true;
            return 0LL;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int64_size(), "int64");
        if (m_was_null) return 0LL;
        return m_next_response.next().table().data_int64(index);
    }
    inline const char *getNumeric(unsigned int col) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1074: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return ""; 
        }
        if (m_types[col].type != NUMERIC) { 
          m_exch->setException("E-UDF-CL-LIB-1075: Wrong input column type, expected NUMERIC, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          m_was_null = true; 
          return ""; 
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "0";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getTimestamp(unsigned int col) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1076: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return ""; 
        }
        if (m_types[col].type != TIMESTAMP) { 
          m_exch->setException("E-UDF-CL-LIB-1077: Wrong input column type, expected TIMESTAMP, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          m_was_null = true; 
          return ""; 
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01 00:00:00.00 0000";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getDate(unsigned int col) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1078: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return ""; 
        }
        if (m_types[col].type != DATE) { 
          m_exch->setException("E-UDF-CL-LIB-1079: Wrong input column type, expected DATE, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          m_was_null = true; 
          return ""; 
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline bool getBoolean(unsigned int col) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1080: Input column "+std::to_string(col)+" does not exist"); 
          m_was_null = true; 
          return ""; 
        }
        if (m_types[col].type != BOOLEAN) { 
          m_exch->setException("E-UDF-CL-LIB-1081: Wrong input column type, expected BOOLEAN, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          m_was_null = true; 
          return ""; 
        }
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
          m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
          m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
          m_message_size(0),
          m_rows_emited(1),
          m_types(*(exaudflib::global.SWIGVM_params_ref->out_types))
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
                    m_exch->setException("F-UDF-CL-LIB-1082: Communication error: failed to serialize data");
                    return;
                }
                zmq::message_t zmsg((void*)m_output_buffer.c_str(), m_output_buffer.length(), NULL, NULL);
                exaudflib_socket_low_level::socket_send(m_socket, zmsg);
                m_emit_request.Clear();
                m_message_size = 0;
            }
            { zmq::message_t zmsg;
                exaudflib_socket_low_level::socket_recv(m_socket, zmsg);
                exascript_response response;
                if (!response.ParseFromArray(zmsg.data(), zmsg.size())) {
                    m_exch->setException("F-UDF-CL-LIB-1083: Communication error: failed to parse data");
                    return;
                }
                if (response.connection_id() != m_connection_id) {
                    std::stringstream sb;
                    sb << "F-UDF-CL-LIB-1084: Received wrong connection id " << response.connection_id()
                       << ", should be " << m_connection_id;
                    m_exch->setException(sb.str().c_str());
                    return;
                }
                if (response.type() == MT_CLOSE) {
                    if (!response.close().has_exception_message())
                        m_exch->setException("F-UDF-CL-LIB-1085: Unknown error occured");
                    else 
                      m_exch->setException(response.close().exception_message().c_str());
                    return;
                }
                if (response.type() != MT_EMIT) {
                    m_exch->setException("F-UDF-CL-LIB-1086: Wrong response type, got "+
                                            msg_conversion::convert_message_type_to_string(response.type()));
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
                m_exch->setException("F-UDF-CL-LIB-1087: Unsupported data type found");
                return false;
            case DOUBLE:
                if (m_rowdata.double_data.find(col) == m_rowdata.double_data.end()) {
                    m_exch->setException("F-UDF-CL-LIB-1088: Not enough double columns emited");
                    return false;
                }
                m_message_size += sizeof(double);
                table->add_data_double(m_rowdata.double_data[col]);
                break;
            case INT32:
                if (m_rowdata.int32_data.find(col) == m_rowdata.int32_data.end()) {
                    m_exch->setException("F-UDF-CL-LIB-1089: Not enough int32 columns emited");
                    return false;
                }
                m_message_size += sizeof(int32_t);
                table->add_data_int32(m_rowdata.int32_data[col]);
                break;
            case INT64:
                if (m_rowdata.int64_data.find(col) == m_rowdata.int64_data.end()) {
                    m_exch->setException("F-UDF-CL-LIB-1090: Not enough int64 columns emited");
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
                    m_exch->setException("F-UDF-CL-LIB-1091: Not enough string columns emited");
                    return false;
                }
                m_message_size += sizeof(int32_t) + m_rowdata.string_data[col].length();
                *table->add_data_string() = m_rowdata.string_data[col];
                break;
            case BOOLEAN:
                if (m_rowdata.bool_data.find(col) == m_rowdata.bool_data.end()) {
                    m_exch->setException("F-UDF-CL-LIB-1092: Not enough boolean columns emited");
                    return false;
                }
                m_message_size += 1;
                table->add_data_bool(m_rowdata.bool_data[col]);
                break;
            default:
                m_exch->setException("F-UDF-CL-LIB-1093: Unknown data type found, got "+
                                        msg_conversion::convert_type_to_string(m_types[col].type));
                return false;
            }
        }
        table->add_row_number(m_table_iterator->get_current_row());
        m_rowdata = rowdata_t();
        if (!table->has_rows()) table->set_rows(1);
        else table->set_rows(table->rows() + 1);
        table->set_rows_in_group(0);
        if (m_message_size >= SWIG_MAX_VAR_DATASIZE) {
            if (exaudflib::global.SWIGVM_params_ref->inp_iter_type == EXACTLY_ONCE && exaudflib::global.SWIGVM_params_ref->out_iter_type == EXACTLY_ONCE)
                exaudflib::global.SWIGVM_params_ref->inp_force_finish = true;
            else this->flush();
        }
        return true;
    }
    inline void setDouble(unsigned int col, const double v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1094: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != DOUBLE) { 
          m_exch->setException("E-UDF-CL-LIB-1095: Wrong output column type, expected DOUBLE, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return;
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.double_data[col] = v;
    }
    inline void setString(unsigned int col, const char *v, size_t l) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1096: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != STRING) { 
          m_exch->setException("E-UDF-CL-LIB-1097: Wrong output column type, expected STRING, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setInt32(unsigned int col, const int32_t v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1098: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != INT32) { 
          m_exch->setException("E-UDF-CL-LIB-1099: Wrong output column type, expected INT32, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.int32_data[col] = v;
    }
    inline void setInt64(unsigned int col, const int64_t v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1100: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != INT64) { 
          m_exch->setException("E-UDF-CL-LIB-1101: Wrong output column type, expected INT64, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.int64_data[col] = v;
    }
    inline void setNumeric(unsigned int col, const char *v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1102: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != NUMERIC) { 
          m_exch->setException("E-UDF-CL-LIB-1103: Wrong output column type, expected NUMERIC, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setTimestamp(unsigned int col, const char *v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1104: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != TIMESTAMP) { 
          m_exch->setException("E-UDF-CL-LIB-1105: Wrong output column type, expected TIMESTAMP, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setDate(unsigned int col, const char *v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1106: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != DATE) { 
          m_exch->setException("E-UDF-CL-LIB-1107: Wrong output column type, expected DATE, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.string_data[col] = v;
    }
    inline void setBoolean(unsigned int col, const bool v) {
        if (col >= m_types.size()) { 
          m_exch->setException("E-UDF-CL-LIB-1108: Output column does "+std::to_string(col)+" not exist"); 
          return; 
        }
        if (m_types[col].type != BOOLEAN) { 
          m_exch->setException("E-UDF-CL-LIB-1109: Wrong output column type, expected BOOLEAN, got "+
                              msg_conversion::convert_type_to_string(m_types[col].type));
          return; 
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.bool_data[col] = v;
    }
    inline void setNull(unsigned int col) {
        m_rowdata.null_data[col] = true;
    }
};

unsigned int handle_error(zmq::socket_t& socket, std::string socket_name, SWIGVM* vm, std::string msg, bool shutdown_vm=false){
    DBG_STREAM_MSG(cerr,"### handle error in '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << msg);
    try{
        if(vm!=nullptr && shutdown_vm){
            vm->exception_msg = "";
            vm->shutdown(); // Calls cleanup
            if (vm->exception_msg.size()>0) {
                PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1110","### Caught error in vm->shutdown '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << vm->exception_msg);
                msg ="F-UDF-CL-LIB-1111: Caught exception\n\n"+msg+"\n\n and caught another exception during cleanup\n\n"+vm->exception_msg;
            }
        } 
        delete_vm(vm);
    }  catch (SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1112","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1113","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1114","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): ");
    }
    try{
        send_close(socket, msg);
        ::sleep(1); // give me a chance to die with my parent process
    }  catch (SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1115","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1116","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1117","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ")");
    }

    try{
        stop_all(socket);
    }  catch (SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1118","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1119","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1120","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "):");
    }
    return 1;
}


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
    assert(exaudflib::global.SWIGVM_params_ref != nullptr);

#ifdef PROTEGRITY_PLUGIN_CLIENT
    stringstream socket_name_ss;
#endif
    string socket_name = argv[1];
    exaudflib_check::init_socket_name_file(argv[1]);

    exaudflib_check::set_remote_client(false);
    my_pid = ::getpid();

    zmq::context_t context(1);

    DBG_COND_FUNC_CALL(cerr, print_args(argc,argv));

    if (socket_name.length() > 4 ) {
#ifdef PROTEGRITY_PLUGIN_CLIENT
        // udf plugins might not have arguments
#else
        if (! ((strcmp(argv[2], "lang=python") == 0)
               || (strcmp(argv[2], "lang=r") == 0)
               || (strcmp(argv[2], "lang=java") == 0)
               || (strcmp(argv[2], "lang=streaming") == 0)
               || (strcmp(argv[2], "lang=benchmark") == 0)) )
        {
            PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1121","Remote VM type '" << argv[2] << "' not supported.");
            return 2;
        }
#endif
    } else {
        PRINT_ERROR_MESSAGE(cerr,"F-UDF-CL-LIB-1122", "socket name '" << socket_name << "' is invalid." );
        abort();
    }

    if (strncmp(exaudflib_check::get_socket_name_str(), "tcp:", 4) == 0) {
        exaudflib_check::set_remote_client(true);
    }

    if (socket_name.length() > 6 && strncmp(exaudflib_check::get_socket_name_str(), "ipc:", 4) == 0)
    {        
#ifdef PROTEGRITY_PLUGIN_CLIENT
/*
    DO NOT REMOVE, required for Exasol 6.2
*/
        if (strncmp(exaudflib_check::get_socket_name_file(), "ipc:///tmp/", 11) == 0) {
            socket_name_ss << "ipc://" << getenv("NSEXEC_TMP_PATH") << '/' << &(exaudflib_check::get_socket_name_file()[11]);
            socket_name = socket_name_ss.str();
            socket_name_str = strdup(socket_name_ss.str().c_str());
            exaudflib_check::init_socket_name_file(socket_name_str);
        }
#endif
        exaudflib_check::init_socket_name_file(&(exaudflib_check::get_socket_name_file()[6]));

    }

    DBG_STREAM_MSG(cerr,"### SWIGVM starting " << argv[0] << " with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): '" << argv[1] << '\'');

    exaudflib_check::start_check_thread();


    int linger_timeout = 0;
    int recv_sock_timeout = 1000;
    int send_sock_timeout = 1000;

    if (exaudflib_check::get_remote_client()) {
        recv_sock_timeout = 10000;
        send_sock_timeout = 5000;
    }

reinit:

    DBGMSG(cerr,"Reinit");
    zmq::socket_t socket(context, ZMQ_REQ);

    socket.setsockopt(ZMQ_LINGER, &linger_timeout, sizeof(linger_timeout));
    socket.setsockopt(ZMQ_RCVTIMEO, &recv_sock_timeout, sizeof(recv_sock_timeout));
    socket.setsockopt(ZMQ_SNDTIMEO, &send_sock_timeout, sizeof(send_sock_timeout));

    if (exaudflib_check::get_remote_client()) socket.bind(exaudflib_check::get_socket_name_str());
    else socket.connect(exaudflib_check::get_socket_name_str());

    exaudflib::global.SWIGVM_params_ref->sock = &socket;
    exaudflib::global.SWIGVM_params_ref->exch = &exaudflib::global.exchandler;

    SWIGVM* vm=nullptr;

    if (!send_init(socket, socket_name)) {
        if (!exaudflib_check::get_remote_client() && exaudflib::global.exchandler.exthrowed) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1123: " +
                                exaudflib::global.exchandler.exmsg, false);
        }else{
            goto reinit;
        }
    }

    exaudflib::global.initSwigParams();

    bool shutdown_vm_in_case_of_error = false;
    try {
        vm = vmMaker();
        if (vm == nullptr) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1124: Unknown or unsupported VM type", false);
        }
        if (vm->exception_msg.size()>0) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1125: "+vm->exception_msg, false);
        }
        shutdown_vm_in_case_of_error = true;
        exaudflib_socket_low_level::init(vm->useZmqSocketLocks());
        if (exaudflib::global.singleCallMode) {
            ExecutionGraph::EmptyDTO noArg; // used as dummy arg
            for (;;) {
                // in single call mode, after MT_RUN from the client,
                // EXASolution responds with a CALL message that specifies
                // the single call function to be made
                if (!send_run(socket)) {
                    break;
                }

                assert(exaudflib::global.singleCallFunction != single_call_function_id_e::SC_FN_NIL);
                try {
                    const char* result = nullptr;
                    switch (exaudflib::global.singleCallFunction)
                    {
                        case single_call_function_id_e::SC_FN_NIL:
                            break;
                        case single_call_function_id_e::SC_FN_DEFAULT_OUTPUT_COLUMNS:
                            result = vm->singleCall(exaudflib::global.singleCallFunction,noArg);
                            break;
                        case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
                            assert(!exaudflib::global.singleCall_ImportSpecificationArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_ImportSpecificationArg);
                            exaudflib::global.singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification();  // delete the last argument
                            break;
                        case single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC:
                            assert(!exaudflib::global.singleCall_ExportSpecificationArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_ExportSpecificationArg);
                            exaudflib::global.singleCall_ExportSpecificationArg = ExecutionGraph::ExportSpecification();  // delete the last argument
                            break;
                        case single_call_function_id_e::SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                            assert(!exaudflib::global.singleCall_StringArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_StringArg);
                            break;
                    }
                    if (vm->exception_msg.size()>0) {
                        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1126: "+vm->exception_msg,true);
                    }

                    if (vm->calledUndefinedSingleCall.size()>0) {
                        send_undefined_call(socket, vm->calledUndefinedSingleCall);
                    } else {
                        send_return(socket,result);
                    }

                    if (!send_done(socket)) {
                        break;
                    }
                } catch(...) {}
            }
        } else {
            for(;;) {
                if (!send_run(socket))
                    break;
                exaudflib::global.SWIGVM_params_ref->inp_force_finish = false;
                while(!vm->run_())
                {
                    if (vm->exception_msg.size()>0) {
                        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1127: "+vm->exception_msg,true);
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
                return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1128: "+vm->exception_msg,false);
            }
        }
        send_finished(socket);
    }  catch (SWIGVM::exception &err) {
        DBG_STREAM_MSG(cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1129: "+std::string(err.what()),shutdown_vm_in_case_of_error);
    } catch (std::exception &err) {
        DBG_STREAM_MSG(cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1130: "+std::string(err.what()),shutdown_vm_in_case_of_error);
    } catch (...) {
        DBG_STREAM_MSG(cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ')');
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1131: Internal/Unknown error",shutdown_vm_in_case_of_error);
    }

    DBG_STREAM_MSG(cerr,"### SWIGVM finishing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ')');

    delete_vm(vm);
    stop_all(socket);
    return 0;
}


} // extern "C"
