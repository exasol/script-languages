#include "exaudflib/impl/swig/swig_meta_data.h"
#include "exaudflib/impl/exaudflib_global.h"

SWIGVMContainers::SWIGMetadata_Impl::SWIGMetadata_Impl():
SWIGMetadata(false),
m_connection_id(exaudflib::global.SWIGVM_params_ref->connection_id),
m_socket(*(exaudflib::global.SWIGVM_params_ref->sock)),
m_exch(exaudflib::global.SWIGVM_params_ref->exch),
m_db_name(exaudflib::global.SWIGVM_params_ref->dbname),
m_db_version(exaudflib::global.SWIGVM_params_ref->dbversion),
m_script_name(exaudflib::global.SWIGVM_params_ref->script_name),
m_script_schema(exaudflib::global.SWIGVM_params_ref->script_schema),
m_current_user(exaudflib::global.SWIGVM_params_ref->current_user),
m_current_schema(exaudflib::global.SWIGVM_params_ref->current_schema),
m_scope_user(exaudflib::global.SWIGVM_params_ref->scope_user),
m_script_code(exaudflib::global.SWIGVM_params_ref->script_code),
m_session_id(exaudflib::global.SWIGVM_params_ref->session_id),
m_statement_id(exaudflib::global.SWIGVM_params_ref->statement_id),
m_node_count(exaudflib::global.SWIGVM_params_ref->node_count),
m_node_id(exaudflib::global.SWIGVM_params_ref->node_id),
m_vm_id(exaudflib::global.SWIGVM_params_ref->vm_id),
m_input_names(*(exaudflib::global.SWIGVM_params_ref->inp_names)),
m_input_types(*(exaudflib::global.SWIGVM_params_ref->inp_types)),
m_input_iter_type(exaudflib::global.SWIGVM_params_ref->inp_iter_type),
m_output_names(*(exaudflib::global.SWIGVM_params_ref->out_names)),
m_output_types(*(exaudflib::global.SWIGVM_params_ref->out_types)),
m_output_iter_type(exaudflib::global.SWIGVM_params_ref->out_iter_type),
m_memory_limit(exaudflib::global.SWIGVM_params_ref->maximal_memory_limit),
m_vm_type(exaudflib::global.SWIGVM_params_ref->vm_type),
m_is_emitted(*(exaudflib::global.SWIGVM_params_ref->is_emitted)),
m_pluginLanguageName(exaudflib::global.SWIGVM_params_ref->pluginName),
m_pluginURI(exaudflib::global.SWIGVM_params_ref->pluginURI),
m_outputAddress(exaudflib::global.SWIGVM_params_ref->outputAddress)
{
    { std::stringstream sb; sb << m_session_id; m_session_id_s = sb.str(); }
    { std::stringstream sb; sb << m_vm_id; m_vm_id_s = sb.str(); }
}
