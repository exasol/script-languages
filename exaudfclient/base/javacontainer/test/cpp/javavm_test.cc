#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include "base/javacontainer/javacontainer_impl.h"
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
    bool useCTPGParser = false;
#ifdef USE_CTPG_PARSER
    useCTPGParser = true;
#endif
    SWIGVMContainers::JavaVMImpl javaVMImpl(false, true, swigFactory, useCTPGParser);
    javaVMInternalStatus.m_exaJavaPath = javaVMImpl.m_exaJavaPath;
    javaVMInternalStatus.m_localClasspath = javaVMImpl.m_localClasspath;
    javaVMInternalStatus.m_scriptCode = javaVMImpl.m_scriptCode;
    javaVMInternalStatus.m_exaJarPath = javaVMImpl.m_exaJarPath;
    javaVMInternalStatus.m_classpath = javaVMImpl.m_classpath;
    javaVMInternalStatus.m_jvmOptions = javaVMImpl.m_jvmOptions;
    javaVMInternalStatus.m_needsCompilation = javaVMImpl.m_needsCompilation;
    delete script_code;
}

