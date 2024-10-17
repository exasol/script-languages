#ifndef JAVACONTAINER_IMPL_H
#define JAVACONTAINER_IMPL_H

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "base/exaudflib/vm/swig_vm.h"
#include <jni.h>

#ifdef ENABLE_JAVA_VM

class JavaVMTest;

namespace SWIGVMContainers {

namespace JavaScriptOptions {
    struct ScriptOptionsParser;
}

class JavaVMImpl {
    public:
        friend class ::JavaVMTest;
        /*
         * scriptOptionsParser: JavaVMImpl takes ownership of ScriptOptionsParser pointer.
         */
        JavaVMImpl(bool checkOnly, bool noJNI,
                    std::unique_ptr<JavaScriptOptions::ScriptOptionsParser> scriptOptionsParser);
        ~JavaVMImpl() {}
        void shutdown();
        bool run();
        const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args,
                                std::string& calledUndefinedSingleCall);
    private:
        void createJvm();
        void addPackageToScript();
        void compileScript();
        bool check(const std::string& errorCode, std::string& calledUndefinedSingleCall); // returns 0 if the check failed
        void registerFunctions();
        void addLocalClasspath();
        bool checkNeedsCompilation();
        void setClasspath();
        void setJvmOptions();
        void addJarToClasspath(const std::string& path);
        void parseScriptOptions(std::unique_ptr<JavaScriptOptions::ScriptOptionsParser> scriptOptionsParser);
        bool m_checkOnly;
        std::string m_exaJavaPath;
        std::string m_localClasspath;
        std::string m_scriptCode;
        std::string m_exaJarPath;
        std::string m_classpath;
        std::vector<std::string> m_jvmOptions;
        JavaVM *m_jvm;
        JNIEnv *m_env;
        bool m_needsCompilation;
};

} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM


#endif //JAVACONTAINER_IMPL_H