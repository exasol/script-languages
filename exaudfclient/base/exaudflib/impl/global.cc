#include "exaudflib/impl/global.h"

namespace exaudflib {
static string g_database_name;
static string g_database_version;
static string g_script_name;
static string g_script_schema;
static string g_current_user;
static string g_scope_user;
static string g_current_schema;
static string g_source_code;
static unsigned long long g_session_id;
static unsigned long g_statement_id;
static unsigned int g_node_count;
static unsigned int g_node_id;
static unsigned long long g_vm_id;

Global global;
}

exaudflib::Global::Global()
: SWIGVM_params_ref(nullptr) {}

void exaudflib::Global::initSwigParams() {
    SWIGVM_params_ref->dbname = (char*) g_database_name.c_str();
    SWIGVM_params_ref->dbversion = (char*) g_database_version.c_str();
    SWIGVM_params_ref->script_name = (char*) g_script_name.c_str();
    SWIGVM_params_ref->script_schema = (char*) g_script_schema.c_str();
    SWIGVM_params_ref->current_user = (char*) g_current_user.c_str();
    SWIGVM_params_ref->current_schema = (char*) g_current_schema.c_str();
    SWIGVM_params_ref->scope_user = (char*) g_scope_user.c_str();
    SWIGVM_params_ref->script_code = (char*) g_source_code.c_str();
    SWIGVM_params_ref->session_id = g_session_id;
    SWIGVM_params_ref->statement_id = g_statement_id;
    SWIGVM_params_ref->node_count = g_node_count;
    SWIGVM_params_ref->node_id = g_node_id;
    SWIGVM_params_ref->vm_id = g_vm_id;
    SWIGVM_params_ref->singleCallMode = singleCallMode;
}

void exaudflib::Global::writeScriptParams(const exascript_info &rep) {
    g_database_name = rep.database_name();
    g_database_version = rep.database_version();
    g_script_name = rep.script_name();
    g_script_schema = rep.script_schema();
    g_current_user = rep.current_user();
    g_scope_user = rep.scope_user();
    if (g_scope_user.size()==0) {         // for backward compatibility when testing with EXASOL 6.0.8 installations at OTTO Brain
        g_scope_user=g_current_user;
    }
    g_current_schema = rep.current_schema();
    g_source_code = rep.source_code();
    g_session_id = rep.session_id();
    g_statement_id = rep.statement_id();
    g_node_count = rep.node_count();
    g_node_id = rep.node_id();
    g_vm_id = rep.vm_id();
}

extern "C" {
    void set_SWIGVM_params(SWIGVMContainers::SWIGVM_params_t* p) {
        printf("Start set SWIGVM params\n");

        exaudflib::global.SWIGVM_params_ref = p;
        printf("End set SWIGVM params\n");
    }
}

