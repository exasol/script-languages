#ifndef SWIG_META_DATA_H
#define SWIG_META_DATA_H

#include "base/exaudflib/load_dynamic.h"
#include "base/exaudflib/swig/swig_common.h"
#include "base/exaudflib/swig/script_data_transfer_objects_wrapper.h"

namespace SWIGVMContainers {

struct SWIGMetadataIf {

        virtual ~SWIGMetadataIf() {};
        virtual const char* databaseName() = 0;
        virtual const char* databaseVersion() = 0;
        virtual const char* scriptName() = 0;
        virtual const char* scriptSchema() = 0;
        virtual const char* currentUser() = 0;
        virtual const char* scopeUser() = 0;
        virtual const char* currentSchema() = 0;
        virtual const char* scriptCode() = 0;
        virtual const unsigned long long sessionID() = 0;
        virtual const char *sessionID_S() = 0;
        virtual const unsigned long statementID() = 0;
        virtual const unsigned int nodeCount() = 0;
        virtual const unsigned int nodeID() = 0;
        virtual const unsigned long long vmID() = 0;
        virtual const unsigned long long memoryLimit() = 0;
        virtual const VMTYPE vmType() = 0;
        virtual const char *vmID_S() = 0;
        virtual const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name) = 0;
        virtual const char* moduleContent(const char* name) = 0;
        virtual const unsigned int inputColumnCount() = 0;
        virtual const char *inputColumnName(unsigned int col) = 0;
        virtual const SWIGVM_datatype_e inputColumnType(unsigned int col) = 0;
        virtual const char *inputColumnTypeName(unsigned int col) = 0;
        virtual const unsigned int inputColumnSize(unsigned int col) = 0;
        virtual const unsigned int inputColumnPrecision(unsigned int col) = 0;
        virtual const unsigned int inputColumnScale(unsigned int col) = 0;
        virtual const SWIGVM_itertype_e inputType() = 0;
        virtual const unsigned int outputColumnCount() = 0;
        virtual const char *outputColumnName(unsigned int col) = 0;
        virtual const SWIGVM_datatype_e outputColumnType(unsigned int col) = 0;
        virtual const char *outputColumnTypeName(unsigned int col) = 0;
        virtual const unsigned int outputColumnSize(unsigned int col) = 0;
        virtual const unsigned int outputColumnPrecision(unsigned int col) = 0;
        virtual const unsigned int outputColumnScale(unsigned int col) = 0;
        virtual const SWIGVM_itertype_e outputType() = 0;
        virtual const bool isEmittedColumn(unsigned int col) = 0;
        virtual const char* checkException() = 0;
        virtual const char* pluginLanguageName() = 0;
        virtual const char* pluginURI() = 0;
        virtual const char* outputAddress() = 0;
};



class SWIGMetadata : public SWIGMetadataIf {
    SWIGMetadataIf* impl=nullptr;
    typedef SWIGVMContainers::SWIGMetadataIf* (*CREATE_METADATA_FUN)();
    public:
        SWIGMetadata()
        {
#ifndef UDF_PLUGIN_CLIENT
            CREATE_METADATA_FUN create = (CREATE_METADATA_FUN)load_dynamic("create_SWIGMetaData");
            impl = create();
#else
            impl = create_SWIGMetaData();
#endif
        }

        virtual ~SWIGMetadata() {
		if (impl!=nullptr) {
        	    delete impl;
	        }
	    }
        virtual const char* databaseName() { return impl->databaseName(); }
        virtual const char* databaseVersion() { return impl->databaseVersion(); }
        virtual const char* scriptName() { return impl->scriptName(); }
        virtual const char* scriptSchema() { return impl->scriptSchema(); }
        virtual const char* currentUser() { return impl->currentUser(); }
        virtual const char* scopeUser() { return impl->scopeUser(); }
        virtual const char* currentSchema() {return impl->currentSchema();}
        virtual const char* scriptCode() { return impl->scriptCode(); }
        virtual const unsigned long long sessionID() { return impl->sessionID(); }
        virtual const char *sessionID_S() { return impl->sessionID_S(); }
        virtual const unsigned long statementID() { return impl->statementID(); }
        virtual const unsigned int nodeCount() { return impl->nodeCount(); }
        virtual const unsigned int nodeID() { return impl->nodeID(); }
        virtual const unsigned long long vmID() { return impl->vmID(); }
        virtual const unsigned long long memoryLimit() { return impl->memoryLimit(); }
        virtual const VMTYPE vmType() { return impl->vmType(); }
        virtual const char *vmID_S() { return impl->vmID_S(); }
        virtual const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name){
            return impl->connectionInformation(connection_name);
        }
        virtual const char* moduleContent(const char* name) {return impl->moduleContent(name);}
        virtual const unsigned int inputColumnCount() { return impl->inputColumnCount(); }
        virtual const char *inputColumnName(unsigned int col) { return impl->inputColumnName(col);}
        virtual const SWIGVM_datatype_e inputColumnType(unsigned int col) { return impl->inputColumnType(col);}
        virtual const char *inputColumnTypeName(unsigned int col) { return impl->inputColumnTypeName(col); }
        virtual const unsigned int inputColumnSize(unsigned int col) {return impl->inputColumnSize(col);}
        virtual const unsigned int inputColumnPrecision(unsigned int col) { return impl->inputColumnPrecision(col); }
        virtual const unsigned int inputColumnScale(unsigned int col) { return impl->inputColumnScale(col);}
        virtual const SWIGVM_itertype_e inputType() { return impl->inputType();}
        virtual const unsigned int outputColumnCount() { return impl->outputColumnCount(); }
        virtual const char *outputColumnName(unsigned int col) { return impl->outputColumnName(col); }
        virtual const SWIGVM_datatype_e outputColumnType(unsigned int col) { return impl->outputColumnType(col); }
        virtual const char *outputColumnTypeName(unsigned int col) { return impl->outputColumnTypeName(col); }
        virtual const unsigned int outputColumnSize(unsigned int col) { return impl->outputColumnSize(col); }
        virtual const unsigned int outputColumnPrecision(unsigned int col) {return impl->outputColumnPrecision(col);}
        virtual const unsigned int outputColumnScale(unsigned int col) {return impl->outputColumnScale(col);}
        virtual const SWIGVM_itertype_e outputType() { return impl->outputType(); }
        virtual const bool isEmittedColumn(unsigned int col){ return impl->isEmittedColumn(col);}
        virtual const char* checkException() {return impl->checkException();}
        virtual const char* pluginLanguageName() {return impl->pluginLanguageName();}
        virtual const char* pluginURI() {return impl->pluginURI();}
        virtual const char* outputAddress() {return impl->outputAddress();}
};

} // namespace SWIGVMContainers

#endif //SWIG_META_DATA_H