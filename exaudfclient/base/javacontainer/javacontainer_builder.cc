#include "base/javacontainer/javacontainer_builder.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/javacontainer/javacontainer.h"
#include "base/swig_factory/swig_factory_impl.h"

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

JavaContainerBuilder::JavaContainerBuilder()
: m_parser() {}

JavaContainerBuilder& JavaContainerBuilder::useCtpgParser(const bool useCtpgParser) {
    m_parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserCTPG>(std::make_unique<SwigFactoryImpl>());
    return *this;
}

JavaVMach* JavaContainerBuilder::build() {
    if (m_parser) {
        return new JavaVMach(false, std::move(m_parser));
    } else {
        m_parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserLegacy>(std::make_unique<SwigFactoryImpl>());
        return new JavaVMach(false, std::move(m_parser));
    }
}


} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM
