#ifndef SCRIPT_DATA_TRANSFER_OBJECT_C_WRAPPER_H
#define SCRIPT_DATA_TRANSFER_OBJECT_C_WRAPPER_H

#ifdef BUILDINSWIGDIR
#include "scriptDTO.h"
#else
#include <engine/exscript/script_data_transfer_objects.h>
#endif

namespace ExecutionGraph
{

class ConnectionInformationWrapper
{
public:
    explicit ConnectionInformationWrapper(const ConnectionInformation& connectionInformation_);
    char* copyKind() const;
    char* copyAddress() const;
    char* copyUser() const;
    char* copyPassword() const;

protected:
    const ConnectionInformation connectionInformation;
};


class ImportSpecificationWrapper
{
public:
    explicit ImportSpecificationWrapper(ExecutionGraph::ImportSpecification* importSpecification_);
    bool isSubselect() const;
    size_t numSubselectColumns() const;
    char* copySubselectColumnName(size_t i) const;
    char* copySubselectColumnType(size_t i) const;
    bool hasConnectionName() const;
    bool hasConnectionInformation() const;
    char* copyConnectionName() const;
    const ConnectionInformationWrapper getConnectionInformation() const;
    size_t getNumberOfParameters() const;
    char* copyKey(size_t pos) const;
    char* copyValue(size_t pos) const;
protected:
    ImportSpecification* importSpecification;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};



}

#endif // SCRIPT_DATA_TRANSFER_OBJECT_C_WRAPPER_H
