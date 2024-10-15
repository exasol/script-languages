#include "base/javacontainer/javacontainer_builder.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/swig_factory/swig_factory_impl.h"

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

JavaContainerBuilder::JavaContainerBuilder()
: m_useCtpgParser(false) {}

JavaContainerBuilder& JavaContainerBuilder::useCtpgParser(const bool useCtpgParser) {
    m_useCtpgParser = useCtpgParser;
    return *this;
}

JavaVMach* JavaContainerBuilder::build() {
    std::unique_ptr<JavaScriptOptions::ScriptOptionsParser> parser;
    if (m_useCtpgParser) {
        parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserCTPG>(std::make_unique<SwigFactoryImpl>());
    } else {
        parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserLegacy>(std::make_unique<SwigFactoryImpl>());
    }
    return new JavaVMach(false, std::move(parser));
}


} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM
