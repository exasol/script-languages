#ifndef SWIG_FACTORY_H
#define SWIG_FACTORY_H 1

namespace SWIGVMContainers {

struct SWIGMetadataIf;

struct SwigFactory {

    virtual SWIGMetadataIf* makeSwigMetadata() = 0;

};


} // namespace SWIGVMContainer

#endif //SWIG_FACTORY_H