#ifndef UDF_PLUGIN_INTERFACE_H
#define UDF_PLUGIN_INTERFACE_H

#if defined(UDF_PLUGIN_CLIENT) || defined(USE_STATIC_SWIG)
namespace SWIGVMContainers {
class SWIGMetadataIf;
class AbstractSWIGTableIterator;
class SWIGRAbstractResultHandler;
class SWIGTableIterator;
}

extern "C" {
SWIGVMContainers::SWIGMetadataIf* create_SWIGMetaData();
SWIGVMContainers::AbstractSWIGTableIterator* create_SWIGTableIterator();
SWIGVMContainers::SWIGRAbstractResultHandler* create_SWIGResultHandler(SWIGVMContainers::SWIGTableIterator* table_iterator);
}
#endif // UDF_PLUGIN_CLIENT

#endif // UDF_PLUGIN_INTERFACE_H
