#include "exaudflib/impl/swig/swig_result_handler.h"

SWIGVMContainers::SWIGResultHandler_Impl::SWIGResultHandler_Impl(SWIGTableIterator* table_iterator)
: m_table_iterator(table_iterator),
m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
m_message_size(0),
m_rows_emited(1),
m_types(*(exaudflib::global.SWIGVM_params_ref->out_types))
{}
