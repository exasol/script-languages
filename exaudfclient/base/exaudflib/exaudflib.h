#ifndef EXAUDFLIB_H
#define EXAUDFLIB_H


void* load_dynamic(const char* name);

#ifdef PROTEGRITY_PLUGIN_CLIENT
namespace SWIGVMContainers {
class SWIGMetadata;
class AbstractSWIGTableIterator;
class SWIGRAbstractResultHandler;
class SWIGTableIterator;
}

extern "C" {
SWIGVMContainers::SWIGMetadata* create_SWIGMetaData();
SWIGVMContainers::AbstractSWIGTableIterator* create_SWIGTableIterator();
SWIGVMContainers::SWIGRAbstractResultHandler* create_SWIGResultHandler(SWIGVMContainers::SWIGTableIterator* table_iterator);
}
#endif


#endif // EXAUDFLIB_H
