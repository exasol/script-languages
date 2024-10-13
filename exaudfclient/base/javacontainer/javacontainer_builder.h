#ifndef JAVACONTAINER_BUILDER_H
#define JAVACONTAINER_BUILDER_H

#include <memory>

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

class JavaVMach;
struct SwigFactory;

namespace JavaScriptOptions {

struct ScriptOptionsParser;

}

class JavaContainerBuilder {
    public:
        JavaContainerBuilder(SwigFactory& swigFactory);

        JavaContainerBuilder& useCtpgParser(const bool value);

        JavaVMach* build();

    private:
        std::unique_ptr<JavaScriptOptions::ScriptOptionsParser> m_parser;
        SwigFactory& m_swigFactory;

};

} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM


#endif //JAVACONTAINER_BUILDER_H