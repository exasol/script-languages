#include "exaudflib/impl/swig/swig_table_iterator.h"
#include "exaudflib/impl/exaudflib_global.h"

SWIGVMContainers::SWIGTableIterator_Impl::SWIGTableIterator_Impl():
m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
m_column_count(exaudflib::global.SWIGVM_params_ref->inp_types->size()),
m_col_offsets(exaudflib::global.SWIGVM_params_ref->inp_types->size()),
m_current_row((uint64_t)-1),
m_rows_completed(0),
m_rows_group_completed(1),
m_was_null(false),
m_types(*(exaudflib::global.SWIGVM_params_ref->inp_types))
{
    receive_next_data(false);
}