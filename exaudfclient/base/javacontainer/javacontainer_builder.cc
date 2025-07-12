#include "base/javacontainer/javacontainer_builder.h"
#include "base/javacontainer/script_options/extractor_impl.h"
#include "base/swig_factory/swig_factory_impl.h"

#ifdef ENABLE_JAVA_VM

namespace SWIGVMContainers {

JavaContainerBuilder::JavaContainerBuilder()
: m_useCtpgParser(false) {}

JavaContainerBuilder& JavaContainerBuilder::useCtpgParser() {
    m_useCtpgParser = true;
    return *this;
}

JavaVMach* JavaContainerBuilder::build() {
    std::unique_ptr<JavaScriptOptions::Extractor> extractor;
    if (m_useCtpgParser) {
        extractor = std::make_unique<JavaScriptOptions::tExtractorV2>(std::make_unique<SwigFactoryImpl>());
    } else {
        extractor = std::make_unique<JavaScriptOptions::tExtractorLegacy>(std::make_unique<SwigFactoryImpl>());
    }
    return new JavaVMach(false, std::move(extractor));
}


} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM
