#ifndef WRAPPER_H
#define WRAPPER_H 
#include <vector>
#include <limits>
#include <iostream>
#include <sstream>
#include <zmq.hpp>
#include "script_client.pb.h"

#include <string>
#include <unistd.h>

#include <map>
#include <typeinfo>
#include <cstdlib>
#include <exception>


namespace UDFClient {





class DTOError : public std::exception
{
public:
    explicit DTOError(const std::string& msg_)
        : msg(msg_)
    {}
    virtual ~DTOError() throw() {}
    virtual const char* what() const throw()
    {
        return msg.c_str();
    }

protected:
        std::string msg;
};



class ScriptDTO
{
public:
    virtual bool isEmpty() const = 0;
};

//!
//!
//!
//!

class EmptyDTO : public ScriptDTO
{
public:
    EmptyDTO() {}

    virtual bool isEmpty() const
    {
        return true;
    }
};


//!
//!
//!
//!

class ConnectionInformation : public ScriptDTO
{
public:
    class Error : public DTOError
    {
    public:
        explicit Error(const std::string& msg)
            : DTOError(msg)
        {}
    };

    ConnectionInformation()
        : kind(""), address(""), user(""), password("")
    {}

    ConnectionInformation(const ConnectionInformation& other)
        : kind(other.getKind()),
          address(other.getAddress()),
          user(other.getUser()),
          password(other.getPassword())
    {}

    ConnectionInformation(const std::string& kind_, const std::string& address_, const std::string& user_, const std::string& password_)
        : kind(kind_),
          address(address_),
          user(user_),
          password(password_)
    {
    }

    ConnectionInformation(const std::string& address_, const std::string& user_, const std::string& password_)
        : kind("password"),
          address(address_),
          user(user_),
          password(password_)
    {}

    virtual bool isEmpty() const
    {
        return false;
    }

    const std::string getKind() const
    {
        return kind;
    }

    const std::string getAddress() const
    {
        return address;
    }

    const std::string getUser() const
    {
        return user;
    }

    const std::string getPassword() const
    {
        return password;
    }

    bool hasData() const
    {
        return kind.size() == 0;
    }

protected:
    std::string kind;
    std::string address;
    std::string user;
    std::string password;

};


class StringDTO : public ScriptDTO
{
public:
    StringDTO()
        : arg("")
    {}

    StringDTO(const std::string& arg_)
        : arg(arg_)
    {}

    virtual bool isEmpty() const
    {
        return false;
    }
    const std::string getArg() const
    {
        return arg;
    }
protected:
    std::string arg;
};


//!
//!
//!
//!
class ImportSpecification : public ScriptDTO
{
private:
    bool isEmptySpec;
public:
    class Error : public DTOError
    {
    public:
        explicit Error(const std::string& msg)
            : DTOError(msg)
        {}
    };

    virtual bool isEmpty() const
    {
        return isEmptySpec;
    }

    explicit ImportSpecification()
        : isEmptySpec(true)
    {}

    explicit ImportSpecification(bool isSubselect__)
        : isEmptySpec(false),
          isSubselect_(isSubselect__),
          subselect_column_names(),
          subselect_column_types(),
          connection_name(""),
          connection_information(),
          parameters()
    {
    }


    void appendSubselectColumnName(const std::string& columnName)
    {
        if (!isSubselect())
        {
            throw Error("import specification error: cannot add column name to non-subselect import specification");
        }
        subselect_column_names.push_back(columnName);
    }

    void appendSubselectColumnType(const std::string& columnType)
    {
        if (!isSubselect())
        {
            throw Error("import specification error: cannot add column type to non-subselect import specification");
        }
        subselect_column_types.push_back(columnType);
    }

    void setConnectionName(const std::string& connectionName_)
    {
        if (hasConnectionName())
        {
            throw Error("import specification error: connection name is set more than once");
        }
        if (hasConnectionInformation())
        {
            throw Error("import specification error: cannot set connection name, because there is already connection information set");
        }
        connection_name = connectionName_;
    }

    void setConnectionInformation(const ConnectionInformation& connectionInformation_)
    {
        if (hasConnectionName())
        {
            throw Error("import specification error: cannot set connection information, because there is already a connection name set");
        }
        if (hasConnectionInformation())
        {
            throw Error("import specification error: cannot set connection information more than once");
        }
        connection_information = connectionInformation_;
    }

    void addParameter(const std::string& key, const std::string& value)
    {
        if (parameters.find(key) != parameters.end())
        {
            std::stringstream sb;
            sb << "import specification error: parameter with name '" << key << "', is set more than once";
            throw Error(sb.str());
        }
        parameters[key] = value;
    }


    bool isSubselect() const
    {
        return isSubselect_;
    }

    bool hasSubselectColumnNames() const
    {
        return subselect_column_names.size()>0;
    }

    bool hasSubselectColumnTypes() const
    {
        return subselect_column_types.size()>0;
    }

    bool hasSubselectColumnSpecification() const
    {
        return hasSubselectColumnNames() || hasSubselectColumnTypes();
    }

    bool hasConnectionName() const
    {
        return connection_name.size()>0;
    }

    bool hasConnectionInformation() const
    {
        return connection_information.hasData() == false;
    }

    bool hasParameters() const
    {
        return parameters.size()>0;
    }

    bool hasConsistentColumns() const
    {
        return (isSubselect() && subselect_column_names.size() == subselect_column_types.size()) || (!isSubselect() && subselect_column_types.size() == 0);
    }

    bool isCompleteImportSubselectSpecification() const
    {
        return hasConsistentColumns() && hasSubselectColumnNames() && hasSubselectColumnTypes();
    }

    bool isCompleteImportIntoTargetTableSpecification() const
    {
        return hasConsistentColumns();
    }


    const std::vector<std::string>& getSubselectColumnNames() const
    {
        if (!isSubselect())
        {
            throw Error("import specification error: cannot get column names of non-subselect import specification");
        }
        return subselect_column_names;
    }

    const std::vector<std::string>& getSubselectColumnTypes() const
    {
        if (!isSubselect())
        {
            throw Error("import specification error: cannot get column types of non-subselect import specification");
        }

        return subselect_column_types;
    }


    const std::string getConnectionName() const
    {
        if (!hasConnectionName())
        {
            throw Error("import specification error: cannot get connection name because it is not set");
        }
        return connection_name;
    }

    const ConnectionInformation getConnectionInformation() const
    {
        if (!hasConnectionInformation())
        {
            throw Error("import specification error: cannot get connection information because it is not set");
        }
        return connection_information;
    }

    const std::map<std::string, std::string>& getParametersgetParameters() const
    {
        return parameters;
    }

protected:
    bool isSubselect_;
    std::vector<std::string> subselect_column_names;
    std::vector<std::string> subselect_column_types;
    std::string connection_name;
    ConnectionInformation connection_information;
    std::map<std::string, std::string> parameters;
};

















///////////////////////////////////////////////////////////////////////////////////


void socket_send(zmq::socket_t &socket, zmq::message_t &zmsg);
bool socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error = false);
#define SWIG_MAX_VAR_DATASIZE 4000000

class LanguagePlugin;

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

struct UDFClientExceptionHolder
{
    UDFClientExceptionHolder() :
        has_exception(false) {}

    ~UDFClientExceptionHolder() {}

    void setException(const char* msg) {
        exception_message = msg;
        has_exception = true;
    }

    std::string exception_message;
    bool has_exception;
};

enum Datatype {
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
enum IteratorType {
    EXACTLY_ONCE = 1,
    MULTIPLE = 2
};
struct ColumnType {
    Datatype type;
    std::string type_name;
    unsigned int len;
    unsigned int prec;
    unsigned int scale;
    ColumnType(const Datatype t, const char *n, const unsigned int l, const unsigned int p, const unsigned int s):
        type(t), type_name(n), len(l), prec(p), scale(s) { }
    ColumnType(const Datatype t, const char *n, const unsigned int l): type(t), type_name(n), len(l), prec(0), scale(0) { }
    ColumnType(const Datatype t, const char *n): type(t), type_name(n), len(0), prec(0), scale(0) { }
    ColumnType(): type(UNSUPPORTED), type_name("UNSUPPORTED"), len(0), prec(0), scale(0) { }
};

class LanguagePlugin {
public:
    struct exception: std::exception {
        exception(const char *reason): m_reason(reason) { }
        virtual ~exception() throw() { }
        const char* what() const throw() { return m_reason.c_str(); }
    private:
        std::string m_reason;
    };
    LanguagePlugin() { }
protected:
    virtual ~LanguagePlugin() { }
public:
    virtual void destroy() {delete this;}
    virtual bool run() = 0;
    virtual std::string singleCall(single_call_function_id fn, const UDFClient::ScriptDTO& args) = 0;
};



struct Metadata {
    uint64_t connection_id;
    zmq::socket_t *sock;
    UDFClientExceptionHolder *exch;
    std::string dbname;
    std::string dbversion;
    std::string script_name;
    std::string script_schema;
    std::string script_code;
    unsigned long long session_id;
    unsigned long statement_id;
    unsigned int node_count;
    unsigned int node_id;
    unsigned long long vm_id;
    VMTYPE vm_type;
    unsigned long long maximal_memory_limit;
    std::vector<std::string> inp_names;
    std::vector<ColumnType> inp_types;
    IteratorType inp_iter_type;
    bool inp_force_finish;
    std::vector<std::string> out_names;
    std::vector<ColumnType> out_types;
    IteratorType out_iter_type;
    std::vector<bool> is_emitted;
    bool singleCallMode;

private:
    std::string output_buffer;
    std::string m_temp_code;

    bool isInternalLanguage;
    std::string pluginName;
    std::string pluginURI;
    std::string outputAddress;

public:

    Metadata():
        connection_id(0), sock(NULL),
        exch(NULL), dbname(), dbversion(), script_name(), script_schema(), script_code(),
        session_id(0), statement_id(0), node_count(0), node_id(0), vm_id(0),
        vm_type(VM_UNSUPPORTED), maximal_memory_limit(0),
        inp_iter_type(MULTIPLE),
        inp_force_finish(false),
        out_iter_type(MULTIPLE),
        singleCallMode(false), isInternalLanguage(true), pluginName(""), pluginURI(""), outputAddress("")
    {

    }

    ~Metadata() { }


    const UDFClient::ConnectionInformation connectionInformation(const char* connection_name)
    {
        exascript_request request;
        request.set_type(MT_IMPORT);
        request.set_connection_id(connection_id);
        exascript_import_req *req = request.mutable_import();
        req->set_script_name(connection_name);
        req->set_kind(PB_IMPORT_CONNECTION_INFORMATION);
        if (!request.SerializeToString(&output_buffer)) {
            throw LanguagePlugin::exception("Communication error: failed to serialize data");
        }
        zmq::message_t zmsg_req((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(*sock, zmsg_req);
        zmq::message_t zmsg_rep;
        socket_recv(*sock, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            throw LanguagePlugin::exception("Communication error: failed to parse data");
        }
        if (response.type() != MT_IMPORT) {
            throw LanguagePlugin::exception("Internal error: wrong message type");
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            throw LanguagePlugin::exception(rep.exception_message().c_str());
        }
        if (!rep.has_connection_information()) {
            throw LanguagePlugin::exception("Internal error: No connection information returned");
        }
        connection_information_rep ci = rep.connection_information();
        return ConnectionInformation(ci.kind(), ci.address(), ci.user(), ci.password());
    }



    const char* moduleContent(const char* name) {
        exascript_request request;
        request.set_type(MT_IMPORT);
        request.set_connection_id(connection_id);
        exascript_import_req *req = request.mutable_import();
        req->set_script_name(name);
        req->set_kind(PB_IMPORT_SCRIPT_CODE);
        if (!request.SerializeToString(&output_buffer)) {
            exch->setException("Communication error: failed to serialize data");
            return NULL;
        }
        zmq::message_t zmsg_req((void*)output_buffer.c_str(), output_buffer.length(), NULL, NULL);
        socket_send(*sock, zmsg_req);
        zmq::message_t zmsg_rep;
        socket_recv(*sock, zmsg_rep);
        exascript_response response;
        if (!response.ParseFromArray(zmsg_rep.data(), zmsg_rep.size())) {
            exch->setException("Communication error: failed to parse data");
            return NULL;
        }
        if (response.type() != MT_IMPORT) {
            exch->setException("Internal error: wrong message type");
            return NULL;
        }
        const exascript_import_rep &rep = response.import();
        if (rep.has_exception_message()) {
            exch->setException(rep.exception_message().c_str());
            return NULL;
        }
        if (!rep.has_source_code()) {
            exch->setException("Internal error: No source code returned");
            return NULL;
        }
        m_temp_code = rep.source_code();
        return m_temp_code.c_str();
    }

    inline const unsigned int inputColumnCount() { return inp_names.size(); }
    inline const char *inputColumnName(unsigned int col)
    { return col >= inp_names.size() ? NULL : inp_names[col].c_str(); }
    inline const Datatype inputColumnType(unsigned int col)
    { return col >= inp_types.size() ? UNSUPPORTED : inp_types[col].type; }
    inline const char *inputColumnTypeName(unsigned int col)
    { return col >= inp_types.size() ? NULL : inp_types[col].type_name.c_str(); }
    inline const unsigned int inputColumnSize(unsigned int col)
    { return col >= inp_types.size() ? 0 : inp_types[col].len; }
    inline const unsigned int inputColumnPrecision(unsigned int col)
    { return col >= inp_types.size() ? 0 : inp_types[col].prec; }
    inline const unsigned int inputColumnScale(unsigned int col)
    { return col >= inp_types.size() ? 0 : inp_types[col].scale; }
    inline const IteratorType inputType() { return inp_iter_type; }
    inline const unsigned int outputColumnCount() { return out_names.size(); }
    inline const char *outputColumnName(unsigned int col) {
        if (out_iter_type == EXACTLY_ONCE && col == 0)
            return "RETURN";
        return col >= out_names.size() ? NULL : out_names[col].c_str();
    }
    inline const Datatype outputColumnType(unsigned int col)
    { return col >= out_types.size() ? UNSUPPORTED : out_types[col].type; }
    inline const char *outputColumnTypeName(unsigned int col)
    { return col >= out_types.size() ? NULL : out_types[col].type_name.c_str(); }
    inline const unsigned int outputColumnSize(unsigned int col)
    { return col >= out_types.size() ? 0 : out_types[col].len; }
    inline const unsigned int outputColumnPrecision(unsigned int col)
    { return col >= out_types.size() ? 0 : out_types[col].prec; }
    inline const unsigned int outputColumnScale(unsigned int col)
    { return col >= out_types.size() ? 0 : out_types[col].scale; }
    inline const IteratorType outputType() { return out_iter_type; }
    inline const bool isEmittedColumn(unsigned int col){
        if (col >= is_emitted.size())
        {
            abort();
        }
        return is_emitted[col];
    }
    inline const char* checkException() {
        if (exch->has_exception) {
            exch->has_exception = false;
            return exch->exception_message.c_str();
        } else return NULL;
    }
};


class Iterator {
protected:
    UDFClientExceptionHolder *m_exch;
public:
    Iterator(UDFClientExceptionHolder *exch): m_exch(exch) { }
    virtual ~Iterator() { }
    inline const char* checkException() {
        if (m_exch->has_exception) {
            m_exch->has_exception = false;
            return m_exch->exception_message.c_str();
        } else return NULL;
    }
};
class InputTable: public Iterator {
private:
    Metadata& meta;
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
        for (std::vector<ColumnType>::const_iterator
             it = meta.inp_types.begin(); it != meta.inp_types.end(); ++it, ++current_column, ++null_index)
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
    InputTable(
            Metadata& meta_
            ):

        Iterator(meta_.exch),
        meta(meta_),
        m_connection_id(meta.connection_id),
        m_socket(*(meta.sock)),
        m_column_count(meta.inp_types.size()),
        m_col_offsets(meta.inp_types.size()),
        m_current_row((uint64_t)-1),
        m_rows_completed(0),
        m_rows_group_completed(1),
        m_was_null(false)
    {
        if (!meta.singleCallMode)
        {
            receive_next_data(false);
        }
    }

    ~InputTable() {}

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
        if (meta.inp_force_finish)
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
        meta.inp_force_finish = false;
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
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0.0; }
        if (meta.inp_types[col].type != DOUBLE) {
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
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (meta.inp_types[col].type != STRING) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return "";
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline int32_t getInt32(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0; }
        if (meta.inp_types[col].type != INT32) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return 0;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int32_size(), "int32");
        if (m_was_null) return 0;
        return m_next_response.next().table().data_int32(index);
    }
    inline int64_t getInt64(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return 0LL; }
        if (meta.inp_types[col].type != INT64) {
            m_exch->setException("Wrong column type");
            m_was_null = true;
            return 0LL;
        }
        ssize_t index = check_value(col, m_next_response.next().table().data_int64_size(), "int64");
        if (m_was_null) return 0LL;
        return m_next_response.next().table().data_int64(index);
    }
    inline const char *getNumeric(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (meta.inp_types[col].type != NUMERIC) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "0";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getTimestamp(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (meta.inp_types[col].type != TIMESTAMP) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01 00:00:00.00 0000";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline const char *getDate(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (meta.inp_types[col].type != DATE) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_string_size(), "string");
        if (m_was_null) return "1970-01-01";
        return m_next_response.next().table().data_string(index).c_str();
    }
    inline bool getBoolean(unsigned int col) {
        if (col >= meta.inp_types.size()) { m_exch->setException("Column does not exist"); m_was_null = true; return ""; }
        if (meta.inp_types[col].type != BOOLEAN) { m_exch->setException("Wrong column type"); m_was_null = true; return ""; }
        ssize_t index = check_value(col, m_next_response.next().table().data_bool_size(), "bool");
        if (m_was_null) return false;
        return m_next_response.next().table().data_bool(index);
    }
    inline bool wasNull() { return m_was_null; }
};


class OutputTable: public Iterator {
private:
    Metadata meta;
    InputTable* m_table_iterator;
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
    const std::vector<ColumnType> &m_types;
public:
    OutputTable(
            Metadata meta_, InputTable* table_iterator
            ):
        Iterator(meta_.exch),
        meta(meta_),
        m_table_iterator(table_iterator),
        m_connection_id(meta.connection_id),
        m_socket(*(meta.sock)),
        m_message_size(0),
        m_rows_emited(1),
        m_types((meta.out_types))
    { }
    ~OutputTable() {
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
            if (meta.inp_iter_type == EXACTLY_ONCE && meta.out_iter_type == EXACTLY_ONCE)
                meta.inp_force_finish = true;
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

struct swig_undefined_single_call_exception: public std::exception
{
    swig_undefined_single_call_exception(const std::string& fn): m_fn(fn) { }
    virtual ~swig_undefined_single_call_exception() throw() { }
    const std::string fn() const {return m_fn;}
    const char* what() const throw() {
        std::stringstream sb;
        sb << "Undefined in UDF: " << m_fn;
        return sb.str().c_str();
    }
private:
    const std::string m_fn;
};

}
#endif
