#ifndef EXAUDFCLIENT_SWIGRESULTHANDLER_H
#define EXAUDFCLIENT_SWIGRESULTHANDLER_H

#include "exaudflib/swig/swig_common.h"
#include "exaudflib/swig/swig_result_handler.h"
#include "exaudflib/impl/swig/swig_general_iterator.h"
#include "exaudflib/impl/global.h"
#include "exaudflib/impl/msg_conversion.h"
#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/zmqcontainer.pb.h"

namespace SWIGVMContainers {

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
    m_socket(*(exaudflib::global.sock)),
    m_message_size(0),
    m_rows_emited(1),
    m_types(*(exaudflib::global.SWIGVM_params_ref->out_types))
    {}

    ~SWIGResultHandler_Impl() {}
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
                exaudflib::socket_low_level::socket_send(m_socket, zmsg);
                m_emit_request.Clear();
                m_message_size = 0;
            }
            { zmq::message_t zmsg;
                exaudflib::socket_low_level::socket_recv(m_socket, zmsg);
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
                    exaudflib::msg_conversion::convert_message_type_to_string(response.type()));
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
                                                        exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
            return;
        }
        m_rowdata.null_data[col] = false;
        m_rowdata.bool_data[col] = v;
    }
    inline void setNull(unsigned int col) {
        m_rowdata.null_data[col] = true;
    }
};

} //namespace SWIGVMContainers

#endif //EXAUDFCLIENT_SWIGRESULTHANDLER_H
