#ifndef SWIG_FACTORY_IMPL_H
#define SWIG_FACTORY_IMPL_H 1

#include "base/swig_factory/swig_factory.h"

namespace SWIGVMContainers {

struct SwigFactoryImpl : public SwigFactory {

    virtual SWIGMetadataIf* makeSwigMetadata() override;

};

} // namespace SWIGVMContainers


#endif //namespace SWIG_FACTORY_IMPL_H