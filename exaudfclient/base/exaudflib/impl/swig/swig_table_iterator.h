#ifndef EXAUDFCLIENT_SWIGTABLEITERATOR_H
#define EXAUDFCLIENT_SWIGTABLEITERATOR_H

#include "exaudflib/swig/swig_common.h"
#include "exaudflib/swig/swig_table_iterator.h"
#include "exaudflib/impl/swig/swig_general_iterator.h"
#include "exaudflib/impl/global.h"
#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/impl/msg_conversion.h"

namespace SWIGVMContainers {

class SWIGTableIterator_Impl : public AbstractSWIGTableIterator, SWIGGeneralIterator {
private:
    const uint64_t m_connection_id;
    zmq::socket_t &m_socket;
    std::string m_output_buffer;
    exascript_request m_request;
    exascript_response m_next_response;

    uint64_t m_rows_received;
    struct values_per_row_t {
        uint64_t strings, bools, int32s, int64s, doubles, binaries;
        values_per_row_t(): strings(0), bools(0), int32s(0), int64s(0), doubles(0) {}
        void reset() { strings = bools = int32s = int64s = doubles = binaries = 0; }
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
                case BINARY: m_col_offsets[current_column] = m_values_per_row.binaries++; break;
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
            exaudflib::socket_low_level::socket_send(m_socket, zmsg);
        } {
            zmq::message_t zmsg;
            exaudflib::socket_low_level::socket_recv(m_socket, zmsg);
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
                exaudflib::msg_conversion::convert_message_type_to_string(m_next_response.type()));
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
        m_socket(*(exaudflib::global.sock)),
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
    ~SWIGTableIterator_Impl() {}
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
            m_was_null = true;
            return "";
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "";
        const std::string &s(m_next_response.next().table().data_string(index));
        if (length != NULL) *length = s.length();
        return s.c_str();
    }
    inline const char *getBinary(unsigned int col, size_t *length = NULL) {
        if (col >= m_types.size()) {
            m_exch->setException("E-UDF-CL-LIB-1068: Input column "+std::to_string(col)+" does not exist");
            m_was_null = true;
            return "";
        }
        if (m_types[col].type != BINARY) {
            m_exch->setException("E-UDF-CL-LIB-1069: Wrong input column type, expected BINARY, got "+
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
            m_was_null = true;
            return "";
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_binary_size(), "binary");
        if (m_was_null) return "";
        if (length != NULL) *length = m_next_response.next().table().data_binary_size();
        return m_next_response.next().table().data_binary(index).data();
    }
    inline int32_t getInt32(unsigned int col) {
        if (col >= m_types.size()) {
            m_exch->setException("E-UDF-CL-LIB-1070: Input column "+std::to_string(col)+" does not exist");
            m_was_null = true;
            return 0;
        }
        if (m_types[col].type != INT32) {
            m_exch->setException("E-UDF-CL-LIB-1071: Wrong input column type, expected INT32, got "+
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
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
            exaudflib::msg_conversion::convert_type_to_string(m_types[col].type));
            m_was_null = true;
            return "";
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_bool_size(), "bool");
        if (m_was_null) return false;
        return m_next_response.next().table().data_bool(index);
    }
    inline bool wasNull() { return m_was_null; }
};

} //namespace SWIGVMContainers

#endif //EXAUDFCLIENT_SWIGTABLEITERATOR_H
