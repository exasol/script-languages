#ifndef JAVACONTAINER_BUILDER_H
#define JAVACONTAINER_BUILDER_H

#include <memory>
#include "javacontainer/javacontainer.h"

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser;

}

class JavaContainerBuilder {
    public:
        JavaContainerBuilder();

        JavaContainerBuilder& useCtpgParser();

        JavaVMach* build();

    private:
        bool m_useCtpgParser;
};

} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM


#endif //JAVACONTAINER_BUILDER_H