#ifdef BUILDINSWIGDIR
#include "scriptDTO.h"
#else
#include <engine/exscript/script_data_transfer_objects.h>
#endif
#include <iostream>
#include <sstream>

namespace ExecutionGraph
{

//////////////////////////////
EmptyDTO::EmptyDTO()
{}

void EmptyDTO::accept(ScriptDTOSerializer& serializer) const
{
    return;
}

bool EmptyDTO::isEmpty() const
{
    return true;
}

//////////////////////////////


void StringDTO::accept(ScriptDTOSerializer& serializer) const
{
    serializer.visit(*this);
}

StringDTO::StringDTO()
    : arg("")
{}

StringDTO::StringDTO(const std::string& arg_)
    : arg(arg_)
{}


const std::string StringDTO::getArg() const
{
    return arg;
}

bool StringDTO::isEmpty() const
{
    return false;
}

//////////////////////////////



void ConnectionInformation::accept(ScriptDTOSerializer& serializer) const
{
    serializer.visit(*this);
}

ConnectionInformation::ConnectionInformation(const std::string& kind_, const std::string& address_, const std::string& user_, const std::string& password_)
    : kind(kind_),
      address(address_),
      user(user_),
      password(password_)
{
}

ConnectionInformation::ConnectionInformation(const std::string& address_, const std::string& user_, const std::string& password_)
    : kind("password"),
      address(address_),
      user(user_),
      password(password_)
{}


ConnectionInformation::ConnectionInformation()
    : kind(""), address(""), user(""), password("")
{}

ConnectionInformation::ConnectionInformation(const ConnectionInformation& other)
    : kind(other.getKind()),
      address(other.getAddress()),
      user(other.getUser()),
      password(other.getPassword())
{}

const std::string ConnectionInformation::getKind() const
{
    return kind;
}

const std::string ConnectionInformation::getAddress() const
{
    return address;
}

const std::string ConnectionInformation::getUser() const
{
    return user;
}

const std::string ConnectionInformation::getPassword() const
{
    return password;
}

bool ConnectionInformation::hasData() const
{
    return kind.size() == 0;
}

bool ConnectionInformation::isEmpty() const
{
    return false;
}

//////////////////////////////


//ImportSpecification::ImportSpecificationError::ImportSpecificationError(const std::string& msg_)
//    :msg(msg_)
//{}

//ImportSpecification::ImportSpecificationError::~ImportSpecificationError() throw() {}



//const char* ImportSpecification::ImportSpecificationError::what() const throw()
//{
//    return msg.c_str();
//}



ImportSpecification::ImportSpecification()
    : isEmptySpec(true)
{}

ImportSpecification::ImportSpecification(bool isSubselect__)
    : isEmptySpec(false),
      isSubselect_(isSubselect__),
      subselect_column_names(),
      subselect_column_types(),
      connection_name(""),
      connection_information(),
      parameters()
{
}


void ImportSpecification::accept(ScriptDTOSerializer& serializer) const
{
    serializer.visit(*this);
}

bool ImportSpecification::isEmpty() const
{
    return isEmptySpec;
}


bool ImportSpecification::isSubselect() const
{
    return isSubselect_;
}
bool ImportSpecification::hasSubselectColumnNames() const
{
    return subselect_column_names.size()>0;
}
bool ImportSpecification::hasSubselectColumnTypes() const
{
    return subselect_column_types.size()>0;
}
bool ImportSpecification::hasSubselectColumnSpecification() const
{
    return hasSubselectColumnNames() || hasSubselectColumnTypes();
}
bool ImportSpecification::hasConnectionName() const
{
    return connection_name.size()>0;
}
bool ImportSpecification::hasConnectionInformation() const
{
    return connection_information.hasData() == false;
}
bool ImportSpecification::hasParameters() const
{
    return parameters.size()>0;
}
bool ImportSpecification::hasConsistentColumns() const
{
    return (isSubselect() && subselect_column_names.size() == subselect_column_types.size()) || (!isSubselect() && subselect_column_types.size() == 0);
}

bool ImportSpecification::isCompleteImportSubselectSpecification() const
{
    return hasConsistentColumns() && hasSubselectColumnNames() && hasSubselectColumnTypes();
}

bool ImportSpecification::isCompleteImportIntoTargetTableSpecification() const
{
    return hasConsistentColumns();
}



const std::vector<std::string>& ImportSpecification::getSubselectColumnNames() const
{
    if (!isSubselect())
    {
        throw Error("import specification error: cannot get column names of non-subselect import specification");
    }
    return subselect_column_names;
}

const std::vector<std::string>& ImportSpecification::getSubselectColumnTypes() const
{
    if (!isSubselect())
    {
        throw Error("import specification error: cannot get column types of non-subselect import specification");
    }

    return subselect_column_types;
}


void ImportSpecification::appendSubselectColumnName(const std::string& columnName)
{
    if (!isSubselect())
    {
        throw Error("import specification error: cannot add column name to non-subselect import specification");
    }
    subselect_column_names.push_back(columnName);
}

void ImportSpecification::appendSubselectColumnType(const std::string& columnType)
{
    if (!isSubselect())
    {
        throw Error("import specification error: cannot add column type to non-subselect import specification");
    }
    subselect_column_types.push_back(columnType);
}

void ImportSpecification::setConnectionName(const std::string& connectionName_)
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

void ImportSpecification::setConnectionInformation(const ConnectionInformation& connectionInformation_)
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



void ImportSpecification::addParameter(const std::string& key, const std::string& value)
{
    if (parameters.find(key) != parameters.end())
    {
        std::stringstream sb;
        sb << "import specification error: parameter with name '" << key << "', is set more than once";
        throw Error(sb.str());
    }
    parameters[key] = value;
}


const std::string ImportSpecification::getConnectionName() const
{
    if (!hasConnectionName())
    {
        throw Error("import specification error: cannot get connection name because it is not set");
    }
    return connection_name;
}

const ConnectionInformation ImportSpecification::getConnectionInformation() const
{
    if (!hasConnectionInformation())
    {
        throw Error("import specification error: cannot get connection information because it is not set");
    }
    return connection_information;
}

const std::map<std::string, std::string>& ImportSpecification::getParameters() const
{
    return parameters;
}

}
