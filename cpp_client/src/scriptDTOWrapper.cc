#ifdef BUILDINSWIGDIR
#include "scriptDTOWrapper.h"
#else
#include <engine/exscript/script_data_transfer_objects_wrapper.h>
#endif

#include<cstring>

namespace ExecutionGraph
{

ConnectionInformationWrapper::ConnectionInformationWrapper(const ConnectionInformation& connectionInformation_)
    : connectionInformation(connectionInformation_)
{}

char* ConnectionInformationWrapper::copyKind() const {
    return strdup(connectionInformation.getKind().c_str());
}
char* ConnectionInformationWrapper::copyAddress() const {return strdup(connectionInformation.getAddress().c_str());}
char* ConnectionInformationWrapper::copyUser() const {return strdup(connectionInformation.getUser().c_str());}
char* ConnectionInformationWrapper::copyPassword() const {return strdup(connectionInformation.getPassword().c_str());}


ImportSpecificationWrapper::ImportSpecificationWrapper(ImportSpecification* importSpecification_)
    : importSpecification(importSpecification_)
{
    if (importSpecification != NULL)
    {
        // extract all the keys and values in some random but fixed order
        for (std::map<std::string,std::string>::const_iterator i = importSpecification->getParameters().begin();
             i != importSpecification->getParameters().end();
             ++i)
        {
            keys.push_back(i->first);
            values.push_back(i->second);
        }
    }
}
bool ImportSpecificationWrapper::isSubselect() const {return importSpecification->isSubselect();}
size_t ImportSpecificationWrapper::numSubselectColumns() const {return importSpecification->getSubselectColumnNames().size();}
char* ImportSpecificationWrapper::copySubselectColumnName(size_t i) const {return ::strdup(importSpecification->getSubselectColumnNames().at(i).c_str());}
char* ImportSpecificationWrapper::copySubselectColumnType(size_t i) const {return ::strdup(importSpecification->getSubselectColumnTypes().at(i).c_str());}
bool ImportSpecificationWrapper::hasConnectionName() const {return importSpecification->hasConnectionName();}
bool ImportSpecificationWrapper::hasConnectionInformation() const {return importSpecification->hasConnectionInformation();}
char* ImportSpecificationWrapper::copyConnectionName() const {return ::strdup(importSpecification->getConnectionName().c_str());}
const ConnectionInformationWrapper ImportSpecificationWrapper::getConnectionInformation() const {
    return ConnectionInformationWrapper(importSpecification->getConnectionInformation());
}

size_t ImportSpecificationWrapper::getNumberOfParameters() const {return keys.size();}
char* ImportSpecificationWrapper::copyKey(size_t pos) const {return ::strdup(keys.at(pos).c_str());}
char* ImportSpecificationWrapper::copyValue(size_t pos) const {return ::strdup(values.at(pos).c_str());}

}
