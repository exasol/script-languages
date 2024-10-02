#ifndef JAVACONTAINER_H
#define JAVACONTAINER_H


#include "base/exaudflib/vm/swig_vm.h"
#include <string.h>

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

class JavaVMImpl;
class SwigFactory;

class JavaVMach: public SWIGVM {
    public:
        JavaVMach(bool checkOnly, SwigFactory& swigFactory, bool useCTPGParser);
        virtual ~JavaVMach() {}
        virtual void shutdown();
        virtual bool run();
        virtual const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
    private:
        JavaVMImpl *m_impl;
};

} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM


#endif //JAVACONTAINER_H