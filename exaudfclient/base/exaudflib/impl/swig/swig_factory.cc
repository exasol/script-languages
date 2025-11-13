#include "exaudflib/impl/swig/swig_meta_data.h"
#include "exaudflib/impl/swig/swig_table_iterator.h"
#include "exaudflib/impl/swig/swig_result_handler.h"

extern "C" {

SWIGVMContainers::SWIGMetadataIf* create_SWIGMetaData() {
    return new SWIGVMContainers::SWIGMetadata_Impl();
}

SWIGVMContainers::AbstractSWIGTableIterator* create_SWIGTableIterator() {
    return new SWIGVMContainers::SWIGTableIterator_Impl();
}

SWIGVMContainers::SWIGRAbstractResultHandler* create_SWIGResultHandler(SWIGVMContainers::SWIGTableIterator* table_iterator) {
    return new SWIGVMContainers::SWIGResultHandler_Impl(table_iterator);
}

} //extern "C"