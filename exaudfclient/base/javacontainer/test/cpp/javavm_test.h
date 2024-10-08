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
    bool m_needsCompilation;
    std::vector<std::string> m_jvmOptions;
};

class SwigFactoryTestImpl;

class JavaVMTest {
    public:
        JavaVMTest(std::string scriptCode);

        JavaVMTest(std::string scriptCode, SwigFactoryTestImpl & swigFactory);

        const JavaVMInternalStatus& getJavaVMInternalStatus() {return javaVMInternalStatus;}

    private:
        void run(std::string scriptCode, SwigFactoryTestImpl & swigFactory);
    private:
        JavaVMInternalStatus javaVMInternalStatus;
};

#endif