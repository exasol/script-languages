#ifndef SCRIPT_DATA_TRANSFER_OBJECTS_H
#define SCRIPT_DATA_TRANSFER_OBJECTS_H

#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <cstdlib>
#include <exception>

namespace ExecutionGraph
{

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


class ScriptDTOSerializer
{
public:
    virtual void visit(const class ScriptDTO&) {std::abort();}
    virtual void visit(const class ImportSpecification&) = 0;
    virtual void visit(const class ExportSpecification&) = 0;
    virtual void visit(const class ConnectionInformation&) = 0;
    virtual void visit(const class StringDTO&) {std::abort();} // will be refactored. Only supported for Proto Serializer
};


//!
//!
//!
//!

class ScriptDTO
{
public:
    virtual void accept(ScriptDTOSerializer& serializer) const = 0;
    virtual bool isEmpty() const = 0;
};

//!
//!
//!
//!

class EmptyDTO : public ScriptDTO
{
public:
    EmptyDTO();

    virtual void accept(ScriptDTOSerializer& serializer) const;
    virtual bool isEmpty() const;
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

    ConnectionInformation();

    ConnectionInformation(const ConnectionInformation& other);

    ConnectionInformation(const std::string& kind,
                          const std::string& address,
                          const std::string& user,
                          const std::string& password);

    ConnectionInformation(const std::string& address,
                          const std::string& user,
                          const std::string& password);

    virtual void accept(ScriptDTOSerializer& serializer) const;
    virtual bool isEmpty() const;

    const std::string getKind() const;
    const std::string getAddress() const;
    const std::string getUser() const;
    const std::string getPassword() const;

    bool hasData() const;

protected:
    std::string kind;
    std::string address;
    std::string user;
    std::string password;

};


class StringDTO : public ScriptDTO
{
public:
    StringDTO();
    StringDTO(const std::string& arg);

    virtual void accept(ScriptDTOSerializer& serializer) const;
    virtual bool isEmpty() const;

    const std::string getArg() const;

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

    virtual void accept(ScriptDTOSerializer& serializer) const;
    virtual bool isEmpty() const;

    explicit ImportSpecification();
    explicit ImportSpecification(bool isSubselect__);

    void appendSubselectColumnName(const std::string& columnName);
    void appendSubselectColumnType(const std::string& columnType);
    void setConnectionName(const std::string& connectionName_);
    void setConnectionInformation(const ConnectionInformation& connectionInformation_);
    void addParameter(const std::string& key, const std::string& value);

    bool isSubselect() const;
    bool hasSubselectColumnNames() const;
    bool hasSubselectColumnTypes() const;
    bool hasSubselectColumnSpecification() const;
    bool hasConnectionName() const;
    bool hasConnectionInformation() const;
    bool hasParameters() const;
    bool hasConsistentColumns() const;
    bool isCompleteImportSubselectSpecification() const;
    bool isCompleteImportIntoTargetTableSpecification() const;

    const std::vector<std::string>& getSubselectColumnNames() const;
    const std::vector<std::string>& getSubselectColumnTypes() const;
    const std::string getConnectionName() const;
    const ConnectionInformation getConnectionInformation() const;
    const std::map<std::string, std::string>& getParameters() const;

protected:
    bool isSubselect_;
    std::vector<std::string> subselect_column_names;
    std::vector<std::string> subselect_column_types;
    std::string connection_name;
    ConnectionInformation connection_information;
    std::map<std::string, std::string> parameters;
};


class ExportSpecification : public ScriptDTO
{
public:
    class Error : public DTOError
    {
    public:
        explicit Error(const std::string& msg)
            : DTOError(msg)
        {}
    };

    virtual void accept(ScriptDTOSerializer& serializer) const;
    virtual bool isEmpty() const;

    explicit ExportSpecification();

    void setConnectionName(const std::string& connectionName_);
    void setConnectionInformation(const ConnectionInformation& connectionInformation_);
    void addParameter(const std::string& key, const std::string& value);
    void setTruncate(const bool truncate_);
    void setReplace(const bool replace_);
    void setCreatedBy(const std::string& createdBy_);
    void addSourceColumnName(const std::string& sourceColumnName_);

    bool hasConnectionName() const;
    bool hasConnectionInformation() const;
    bool hasParameters() const;
    bool hasTruncate() const;
    bool hasReplace() const;
    bool hasCreatedBy() const;

    const std::string getConnectionName() const;
    const ConnectionInformation getConnectionInformation() const;
    const std::map<std::string, std::string>& getParameters() const;
    const std::string getCreatedBy() const;
    const std::vector<std::string>& getSourceColumnNames() const;

private:
    bool isEmptySpec;

protected:
    std::string connection_name;
    ConnectionInformation connection_information;
    std::map<std::string, std::string> parameters;
    bool truncate;
    bool replace;
    std::string created_by;
    std::vector<std::string> source_column_names;
};


}

#endif // SCRIPT_DATA_TRANSFER_OBJECTS_H
