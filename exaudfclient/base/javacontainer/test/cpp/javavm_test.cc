#include "javacontainer/test/cpp/javavm_test.h"
#include "javacontainer/test/cpp/swig_factory_test.h"
#include "javacontainer/javacontainer_impl.h"
#include "javacontainer/script_options/extractor_impl.h"
#include <string.h>

std::unique_ptr<SwigFactoryTestImpl> makeDefaultSwigFactory() {
    return std::make_unique<SwigFactoryTestImpl>();
}

JavaVMTest::JavaVMTest(std::string scriptCode) : javaVMInternalStatus() {
    run(scriptCode, std::move(makeDefaultSwigFactory()));
}

JavaVMTest::JavaVMTest(std::string scriptCode, std::unique_ptr<SwigFactoryTestImpl> swigFactory) : javaVMInternalStatus() {
    run(scriptCode, std::move(swigFactory));
}

void JavaVMTest::run(std::string scriptCode, std::unique_ptr<SwigFactoryTestImpl> swigFactory) {
    SWIGVMContainers::SWIGVM_params->script_code = scriptCode.data();
#ifndef USE_EXTRACTOR_V2
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::tExtractorLegacy> extractor =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::tExtractorLegacy>(std::move(swigFactory));
#else
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::tExtractorV2> extractor =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::tExtractorV2>(std::move(swigFactory));
#endif
    SWIGVMContainers::JavaVMImpl javaVMImpl(false, true, std::move(extractor));
    javaVMInternalStatus.m_exaJavaPath = javaVMImpl.m_exaJavaPath;
    javaVMInternalStatus.m_localClasspath = javaVMImpl.m_localClasspath;
    javaVMInternalStatus.m_scriptCode = javaVMImpl.m_scriptCode;
    javaVMInternalStatus.m_exaJarPath = javaVMImpl.m_exaJarPath;
    javaVMInternalStatus.m_classpath = javaVMImpl.m_classpath;
    javaVMInternalStatus.m_jvmOptions = javaVMImpl.m_jvmOptions;
    javaVMInternalStatus.m_needsCompilation = javaVMImpl.m_needsCompilation;
}

