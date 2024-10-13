#include "base/javacontainer/javacontainer_builder.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/javacontainer/javacontainer.h"

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

JavaContainerBuilder::JavaContainerBuilder(SwigFactory& swigFactory)
: m_parser()
, m_swigFactory(swigFactory) {}

JavaContainerBuilder& JavaContainerBuilder::useCtpgParser(const bool useCtpgParser) {
    m_parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserCTPG>();
    return *this;
}

JavaVMach* JavaContainerBuilder::build() {
    if (m_parser) {
        return new JavaVMach(false, m_swigFactory, std::move(m_parser));
    } else {
        m_parser = std::make_unique<JavaScriptOptions::ScriptOptionLinesParserLegacy>();
        return new JavaVMach(false, m_swigFactory, std::move(m_parser));
    }
}


} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM
