#ifndef SWIG_FACTORY_TEST_H
#define SWIG_FACTORY_TEST_H 1

#include <string>
#include <map>
#include "swig_factory/swig_factory.h"
#include <functional>

struct SwigFactoryTestImpl : public SWIGVMContainers::SwigFactory {

    SwigFactoryTestImpl(std::function<const char*(const char*)> callback);

    virtual SWIGVMContainers::SWIGMetadataIf* makeSwigMetadata() override;

private:
    std::function<const char*(const char*)> m_callback;
};


#endif //namespace SWIG_FACTORY_TEST_H