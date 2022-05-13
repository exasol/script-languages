#ifndef EXAUDFCLIENT_EXUDFLIB_GLOBAL_H
#define EXAUDFCLIENT_EXUDFLIB_GLOBAL_H

#include "exaudflib/swig/script_data_transfer_objects.h"
#include "exaudflib/zmqcontainer.pb.h"
#include "exaudflib/swig/swig_common.h"
#include "exaudflib/impl/swig/swig_parameter.h"

namespace exaudflib {
    struct Global {
        Global();
        void initSwigParams();
        void writeScriptParams(const exascript_info &rep);
        SWIGVMContainers::SWIGVM_params_t * SWIGVM_params_ref;
        ExecutionGraph::StringDTO singleCall_StringArg;
        bool singleCallMode;
        SWIGVMContainers::single_call_function_id_e singleCallFunction;
        ExecutionGraph::ImportSpecification singleCall_ImportSpecificationArg;
        ExecutionGraph::ExportSpecification singleCall_ExportSpecificationArg;
        SWIGVMContainers::SWIGVMExceptionHandler exchandler;
    };

    extern Global global;
}


#endif //EXAUDFCLIENT_EXUDFLIB_GLOBAL_H
