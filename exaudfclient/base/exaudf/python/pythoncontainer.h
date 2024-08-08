#ifndef PYTHONVM_H
#define PYTHONVM_H

#include "exaudf/exaudflib/vm/swig_vm.h"

#ifdef ENABLE_PYTHON_VM

namespace SWIGVMContainers {

class PythonVMImpl;

class PythonVM: public SWIGVM {
    public:
        PythonVM(bool checkOnly);
        virtual ~PythonVM() {};
        virtual void shutdown();
        virtual bool run();
        virtual const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
    private:
        PythonVMImpl *m_impl;
};

} //namespace SWIGVMContainers

#endif //ENABLE_PYTHON_VM

#endif //PYTHONVM_H