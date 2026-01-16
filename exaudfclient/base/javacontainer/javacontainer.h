#ifndef JAVACONTAINER_H
#define JAVACONTAINER_H


#include "exaudflib/vm/swig_vm.h"
#include <string.h>
#include <memory>

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

class JavaVMImpl;

namespace JavaScriptOptions {

struct Extractor;

}

class JavaVMach: public SWIGVM {
    public:
        /*
         * scriptOptionsParser: JavaVMach takes ownership of ScriptOptionsParser pointer.
         */
        JavaVMach(bool checkOnly, std::unique_ptr<JavaScriptOptions::Extractor> extractor);
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