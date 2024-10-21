#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include "base/javacontainer/javacontainer_impl.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
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
    std::unique_ptr<char*> script_code = ::strdup(scriptCode.c_str());
    SWIGVMContainers::SWIGVM_params->script_code = script_code.get();
#ifndef USE_CTPG_PARSER
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserLegacy> parser =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserLegacy>(std::move(swigFactory));
#else
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserCTPG> parser =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserCTPG>(std::move(swigFactory));
#endif
    SWIGVMContainers::JavaVMImpl javaVMImpl(false, true, std::move(parser));
    javaVMInternalStatus.m_exaJavaPath = javaVMImpl.m_exaJavaPath;
    javaVMInternalStatus.m_localClasspath = javaVMImpl.m_localClasspath;
    javaVMInternalStatus.m_scriptCode = javaVMImpl.m_scriptCode;
    javaVMInternalStatus.m_exaJarPath = javaVMImpl.m_exaJarPath;
    javaVMInternalStatus.m_classpath = javaVMImpl.m_classpath;
    javaVMInternalStatus.m_jvmOptions = javaVMImpl.m_jvmOptions;
    javaVMInternalStatus.m_needsCompilation = javaVMImpl.m_needsCompilation;
}

