//#ifdef BUILDINSWIGDIR
//#include "scriptDTOWrapper.h"
//#else
#include "script_data_transfer_objects_wrapper.h"
//#endif

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


ExportSpecificationWrapper::ExportSpecificationWrapper(ExportSpecification* exportSpecification_)
    : exportSpecification(exportSpecification_)
{
    if (exportSpecification != NULL)
    {
        // extract all the keys and values in some random but fixed order
        for (std::map<std::string,std::string>::const_iterator i = exportSpecification->getParameters().begin();
             i != exportSpecification->getParameters().end();
             ++i)
        {
            keys.push_back(i->first);
            values.push_back(i->second);
        }

        for (std::vector<std::string>::const_iterator i = exportSpecification->getSourceColumnNames().begin();
             i != exportSpecification->getSourceColumnNames().end(); i++)
        {
            columnNames.push_back(*i);
        }
    }
}
bool ExportSpecificationWrapper::hasConnectionName() const {return exportSpecification->hasConnectionName();}
bool ExportSpecificationWrapper::hasConnectionInformation() const {return exportSpecification->hasConnectionInformation();}
char* ExportSpecificationWrapper::copyConnectionName() const {return ::strdup(exportSpecification->getConnectionName().c_str());}
const ConnectionInformationWrapper ExportSpecificationWrapper::getConnectionInformation() const {
    return ConnectionInformationWrapper(exportSpecification->getConnectionInformation());
}
size_t ExportSpecificationWrapper::getNumberOfParameters() const {return keys.size();}
char* ExportSpecificationWrapper::copyKey(size_t pos) const {return ::strdup(keys.at(pos).c_str());}
char* ExportSpecificationWrapper::copyValue(size_t pos) const {return ::strdup(values.at(pos).c_str());}
bool ExportSpecificationWrapper::hasTruncate() const {return exportSpecification->hasTruncate();}
bool ExportSpecificationWrapper::hasReplace() const {return exportSpecification->hasReplace();}
bool ExportSpecificationWrapper::hasCreatedBy() const {return exportSpecification->hasCreatedBy();}
char* ExportSpecificationWrapper::copyCreatedBy() const {
    if (exportSpecification->getCreatedBy().empty())
        return 0;
    else
        return ::strdup(exportSpecification->getCreatedBy().c_str());
}

size_t ExportSpecificationWrapper::numSourceColumns() const {return columnNames.size();}
char* ExportSpecificationWrapper::copySourceColumnName(size_t pos) const {
    if (columnNames.at(pos).empty())
        return 0;
    else
        return ::strdup(columnNames.at(pos).c_str());
}

}
