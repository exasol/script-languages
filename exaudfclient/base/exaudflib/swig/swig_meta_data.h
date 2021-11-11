#ifndef SWIG_META_DATA_H
#define SWIG_META_DATA_H

#include "exaudflib/exaudflib.h"
#include "exaudflib/swig/swig_common.h"
#include "exaudflib/script_data_transfer_objects_wrapper.h"

namespace SWIGVMContainers {

class SWIGMetadata {
    SWIGMetadata* impl=nullptr;
    typedef SWIGVMContainers::SWIGMetadata* (*CREATE_METADATA_FUN)();
    public:
        SWIGMetadata()
        {
#ifndef PROTEGRITY_PLUGIN_CLIENT
            CREATE_METADATA_FUN create = (CREATE_METADATA_FUN)load_dynamic("create_SWIGMetaData");
            impl = create();
#else
            impl = create_SWIGMetaData();
#endif
        }
        /* hack: use this constructor to avoid cycling loading of this class */
        SWIGMetadata(bool) {}

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