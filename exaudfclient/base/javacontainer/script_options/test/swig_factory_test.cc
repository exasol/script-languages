#include "javacontainer/script_options/test/swig_factory_test.h"
#include "exaudflib/swig/swig_meta_data.h"

#include <stdexcept>

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
    NotImplemented(const std::string funcName) : std::logic_error("Function " + funcName + " not yet implemented") { };
};

class SWIGMetadataTest : public SWIGVMContainers::SWIGMetadataIf {

public:
    SWIGMetadataTest(std::function<const char*(const char*)> callback): m_callback(callback)  {}
    virtual const char* databaseName() { throw NotImplemented("databaseName"); return nullptr;}
    virtual const char* databaseVersion() { throw NotImplemented("databaseVersion"); return nullptr;}
    virtual const char* scriptName() { throw NotImplemented("scriptName"); return nullptr;}
    virtual const char* scriptSchema() { throw NotImplemented("scriptSchema"); return nullptr;}
    virtual const char* currentUser() { throw NotImplemented("currentUser"); return nullptr;}
    virtual const char* scopeUser() { throw NotImplemented("scopeUser"); return nullptr;}
    virtual const char* currentSchema() { throw NotImplemented("currentSchema"); return nullptr;}
    virtual const char* scriptCode() { throw NotImplemented("scriptCode"); return nullptr;}
    virtual const unsigned long long sessionID() { throw NotImplemented("sessionID"); return 0;}
    virtual const char *sessionID_S() { throw NotImplemented("sessionID_S"); return nullptr;}
    virtual const unsigned long statementID() { throw NotImplemented("statementID"); return 0;}
    virtual const unsigned int nodeCount() { throw NotImplemented("nodeCount"); return 0;}
    virtual const unsigned int nodeID() { throw NotImplemented("nodeID"); return 0;}
    virtual const unsigned long long vmID() { throw NotImplemented("vmID"); return 0;}
    virtual const unsigned long long memoryLimit() { throw NotImplemented("memoryLimit"); return 0;}
    virtual const SWIGVMContainers::VMTYPE vmType() {
        throw NotImplemented("vmType");
        return SWIGVMContainers::VM_UNSUPPORTED;
    }
    virtual const char *vmID_S() { throw NotImplemented("vmID_S"); return nullptr;}
    virtual const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name) {
        throw NotImplemented("connectionInformation"); return nullptr;
    }
    virtual const char* moduleContent(const char* name) {
        return m_callback(name);
    }
    virtual const unsigned int inputColumnCount() { throw NotImplemented("inputColumnCount"); return 0;}
    virtual const char *inputColumnName(unsigned int col) {
        throw NotImplemented("inputColumnName");
        return nullptr;
    }
    virtual const SWIGVMContainers::SWIGVM_datatype_e inputColumnType(unsigned int col) {
        throw NotImplemented("inputColumnType");
        return SWIGVMContainers::UNSUPPORTED;
    }
    virtual const char *inputColumnTypeName(unsigned int col) {
        throw NotImplemented("inputColumnTypeName"); return nullptr;
    }
    virtual const unsigned int inputColumnSize(unsigned int col) {
        throw NotImplemented("inputColumnSize"); return 0;
    }
    virtual const unsigned int inputColumnPrecision(unsigned int col) {
        throw NotImplemented("inputColumnPrecision"); return 0;
    }
    virtual const unsigned int inputColumnScale(unsigned int col) {
        throw NotImplemented("inputColumnScale"); return 0;
    }
    virtual const SWIGVMContainers::SWIGVM_itertype_e inputType() {
        throw NotImplemented("inputType");
        return SWIGVMContainers::EXACTLY_ONCE;
    }
    virtual const unsigned int outputColumnCount() { throw NotImplemented("outputColumnCount"); return 0;}
    virtual const char *outputColumnName(unsigned int col) { throw NotImplemented("outputColumnName"); return nullptr;}
    virtual const SWIGVMContainers::SWIGVM_datatype_e outputColumnType(unsigned int col) {
        throw NotImplemented("outputColumnType");
        return SWIGVMContainers::UNSUPPORTED;
    }
    virtual const char *outputColumnTypeName(unsigned int col) {
        throw NotImplemented("outputColumnTypeName"); return nullptr;
    }
    virtual const unsigned int outputColumnSize(unsigned int col) {
        throw NotImplemented("outputColumnSize"); return 0;
    }
    virtual const unsigned int outputColumnPrecision(unsigned int col) {
        throw NotImplemented("outputColumnPrecision"); return 0;
    }
    virtual const unsigned int outputColumnScale(unsigned int col) {
        throw NotImplemented("outputColumnScale"); return 0;
    }
    virtual const SWIGVMContainers::SWIGVM_itertype_e outputType() {
        throw NotImplemented("outputType");
        return SWIGVMContainers::EXACTLY_ONCE;
    }
    virtual const bool isEmittedColumn(unsigned int col) { throw NotImplemented("isEmittedColumn"); return false;}
    virtual const char* checkException() { return nullptr;}
    virtual const char* pluginLanguageName() { throw NotImplemented("pluginLanguageName"); return nullptr;}
    virtual const char* pluginURI() { throw NotImplemented("pluginURI"); return nullptr;}
    virtual const char* outputAddress() { throw NotImplemented("outputAddress"); return nullptr;}

private:
    std::function<const char*(const char*)> m_callback;
};

SwigFactoryTestImpl::SwigFactoryTestImpl(std::function<const char*(const char*)> callback)
: m_callback(callback) {}

SWIGVMContainers::SWIGMetadataIf* SwigFactoryTestImpl::makeSwigMetadata() {
    return new SWIGMetadataTest(m_callback);
}
