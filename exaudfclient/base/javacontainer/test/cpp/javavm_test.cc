#include "javacontainer/test/cpp/javavm_test.h"
#include "javacontainer/javacontainer.h"
#include "exaudflib/exaudflib.h"


JavaVMTest::JavaVMTest(std::string scriptCode) : javaVMInternalStatus() {
    char* script_code = ::strdup(scriptCode.c_str());
    SWIGVMContainers::SWIGVM_params->script_code = script_code;
    SWIGVMContainers::JavaVMImpl javaVMImpl(false, true);
    javaVMInternalStatus.m_exaJavaPath = javaVMImpl.m_exaJavaPath;
    javaVMInternalStatus.m_localClasspath = javaVMImpl.m_localClasspath;
    javaVMInternalStatus.m_scriptCode = javaVMImpl.m_scriptCode;
    javaVMInternalStatus.m_exaJarPath = javaVMImpl.m_exaJarPath;
    javaVMInternalStatus.m_classpath = javaVMImpl.m_classpath;
    javaVMInternalStatus.m_jarPaths = javaVMImpl.m_jarPaths;
    javaVMInternalStatus.m_jvmOptions = javaVMImpl.m_jvmOptions;
    delete script_code;
}


