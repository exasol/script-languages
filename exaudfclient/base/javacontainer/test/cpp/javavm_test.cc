#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include "base/javacontainer/javacontainer_impl.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include <string.h>


SwigFactoryTestImpl & defaultSwigFactory() {
    static SwigFactoryTestImpl swigFactory;
    return swigFactory;
}

JavaVMTest::JavaVMTest(std::string scriptCode) : javaVMInternalStatus() {
    run(scriptCode, defaultSwigFactory());
}

JavaVMTest::JavaVMTest(std::string scriptCode, SwigFactoryTestImpl & swigFactory) : javaVMInternalStatus() {
    run(scriptCode, swigFactory);
}

void JavaVMTest::run(std::string scriptCode, SwigFactoryTestImpl & swigFactory) {
    char* script_code = ::strdup(scriptCode.c_str());
    SWIGVMContainers::SWIGVM_params->script_code = script_code;
#ifndef USE_CTPG_PARSER
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserLegacy> parser =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserLegacy>();
#else
    std::unique_ptr<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserCTPG> parser =
         std::make_unique<SWIGVMContainers::JavaScriptOptions::ScriptOptionLinesParserCTPG>();
#endif
    SWIGVMContainers::JavaVMImpl javaVMImpl(false, true, swigFactory, std::move(parser));
    javaVMInternalStatus.m_exaJavaPath = javaVMImpl.m_exaJavaPath;
    javaVMInternalStatus.m_localClasspath = javaVMImpl.m_localClasspath;
    javaVMInternalStatus.m_scriptCode = javaVMImpl.m_scriptCode;
    javaVMInternalStatus.m_exaJarPath = javaVMImpl.m_exaJarPath;
    javaVMInternalStatus.m_classpath = javaVMImpl.m_classpath;
    javaVMInternalStatus.m_jvmOptions = javaVMImpl.m_jvmOptions;
    javaVMInternalStatus.m_needsCompilation = javaVMImpl.m_needsCompilation;
    ::free(script_code);
}

