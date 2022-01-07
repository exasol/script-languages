#ifndef JAVA_VM_TEST
#define JAVA_VM_TEST

#include <string>
#include <set>
#include <vector>

struct JavaVMInternalStatus {
    std::string m_exaJavaPath;
    std::string m_localClasspath;
    std::string m_scriptCode;
    std::string m_exaJarPath;
    std::string m_classpath;
    std::set<std::string> m_jarPaths;
    std::vector<std::string> m_jvmOptions;
};

class JavaVMTest {
    public:
        JavaVMTest(std::string scriptCode);

        const JavaVMInternalStatus& getJavaVMInternalStatus() {return javaVMInternalStatus;}

    private:
        JavaVMInternalStatus javaVMInternalStatus;
};

#endif