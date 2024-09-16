#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include "base/exaudflib/swig/swig_meta_data.h"

#include <stdexcept>

class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};

class SWIGMetadataTest : public SWIGVMContainers::SWIGMetadataIf {

public:
    SWIGMetadataTest(const std::map<std::string, std::string> & moduleContent) : m_moduleContent(moduleContent) {};
    virtual const char* databaseName() { throw NotImplemented(); return nullptr;}
    virtual const char* databaseVersion() { throw NotImplemented(); return nullptr;}
    virtual const char* scriptName() { throw NotImplemented(); return nullptr;}
    virtual const char* scriptSchema() { throw NotImplemented(); return nullptr;}
    virtual const char* currentUser() { throw NotImplemented(); return nullptr;}
    virtual const char* scopeUser() { throw NotImplemented(); return nullptr;}
    virtual const char* currentSchema() { throw NotImplemented(); return nullptr;}
    virtual const char* scriptCode() { throw NotImplemented(); return nullptr;}
    virtual const unsigned long long sessionID() { throw NotImplemented(); return 0;}
    virtual const char *sessionID_S() { throw NotImplemented(); return nullptr;}
    virtual const unsigned long statementID() { throw NotImplemented(); return 0;}
    virtual const unsigned int nodeCount() { throw NotImplemented(); return 0;}
    virtual const unsigned int nodeID() { throw NotImplemented(); return 0;}
    virtual const unsigned long long vmID() { throw NotImplemented(); return 0;}
    virtual const unsigned long long memoryLimit() { throw NotImplemented(); return 0;}
    virtual const SWIGVMContainers::VMTYPE vmType() { throw NotImplemented(); return SWIGVMContainers::VM_UNSUPPORTED;}
    virtual const char *vmID_S() { throw NotImplemented(); return nullptr;}
    virtual const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name) {
        throw NotImplemented(); return nullptr;
    }
    virtual const char* moduleContent(const char* name) {
        auto it = m_moduleContent.find(std::string(name));
        if (m_moduleContent.end() == it) {
            throw std::invalid_argument("Script not found.");
        }
        return it->second.c_str();
    }
    virtual const unsigned int inputColumnCount() { throw NotImplemented(); return 0;}
    virtual const char *inputColumnName(unsigned int col) { throw NotImplemented(); return nullptr;}
    virtual const SWIGVMContainers::SWIGVM_datatype_e inputColumnType(unsigned int col) {
        throw NotImplemented();
        return SWIGVMContainers::UNSUPPORTED;
    }
    virtual const char *inputColumnTypeName(unsigned int col) { throw NotImplemented(); return nullptr;}
    virtual const unsigned int inputColumnSize(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const unsigned int inputColumnPrecision(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const unsigned int inputColumnScale(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const SWIGVMContainers::SWIGVM_itertype_e inputType() {
        throw NotImplemented();
        return SWIGVMContainers::EXACTLY_ONCE;
    }
    virtual const unsigned int outputColumnCount() { throw NotImplemented(); return 0;}
    virtual const char *outputColumnName(unsigned int col) { throw NotImplemented(); return nullptr;}
    virtual const SWIGVMContainers::SWIGVM_datatype_e outputColumnType(unsigned int col) {
        throw NotImplemented();
        return SWIGVMContainers::UNSUPPORTED;
    }
    virtual const char *outputColumnTypeName(unsigned int col) { throw NotImplemented(); return nullptr;}
    virtual const unsigned int outputColumnSize(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const unsigned int outputColumnPrecision(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const unsigned int outputColumnScale(unsigned int col) { throw NotImplemented(); return 0;}
    virtual const SWIGVMContainers::SWIGVM_itertype_e outputType() {
        throw NotImplemented();
        return SWIGVMContainers::EXACTLY_ONCE;
    }
    virtual const bool isEmittedColumn(unsigned int col) { throw NotImplemented(); return false;}
    virtual const char* checkException() { throw NotImplemented(); return nullptr;}
    virtual const char* pluginLanguageName() { throw NotImplemented(); return nullptr;}
    virtual const char* pluginURI() { throw NotImplemented(); return nullptr;}
    virtual const char* outputAddress() { throw NotImplemented(); return nullptr;}

private:

    std::map<std::string, std::string> m_moduleContent;
};



SwigFactoryTestImpl::SwigFactoryTestImpl() {}


void SwigFactoryTestImpl::addModule(const std::string key, const std::string script) {
    m_moduleContent.insert(std::make_pair(key, script));
}

SWIGVMContainers::SWIGMetadataIf* SwigFactoryTestImpl::makeSwigMetadata() {
    return new SWIGMetadataTest(m_moduleContent);
}
