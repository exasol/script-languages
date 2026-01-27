#include "swig_factory/swig_factory_impl.h"
#include "exaudflib/load_dynamic.h"
#include "exaudflib/swig/swig_common.h"

namespace SWIGVMContainers {

SWIGMetadataIf* SwigFactoryImpl::makeSwigMetadata() {
    SWIGMetadataIf* impl=nullptr;
    typedef SWIGVMContainers::SWIGMetadataIf* (*CREATE_METADATA_FUN)();
#ifndef UDF_PLUGIN_CLIENT
    CREATE_METADATA_FUN create = (CREATE_METADATA_FUN)load_dynamic("create_SWIGMetaData");
    impl = create();
#else
    impl = create_SWIGMetaData();
#endif
    return impl;
}

} //namespace SWIGVMContainers
