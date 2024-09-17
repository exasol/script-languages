#ifndef SWIG_FACTORY_TEST_H
#define SWIG_FACTORY_TEST_H 1

#include <string>
#include <map>
#include "base/swig_factory/swig_factory.h"

struct SwigFactoryTestImpl : public SWIGVMContainers::SwigFactory {

    SwigFactoryTestImpl();

    void addModule(const std::string key, const std::string script);

    virtual SWIGVMContainers::SWIGMetadataIf* makeSwigMetadata() override;

private:
    std::map<std::string, std::string> m_moduleContent;
};


#endif //namespace SWIG_FACTORY_TEST_H