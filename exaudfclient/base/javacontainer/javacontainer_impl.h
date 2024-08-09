#ifndef JAVACONTAINER_IMPL_H
#define JAVACONTAINER_IMPL_H

#include <string>
#include <vector>
#include <set>

#include "base/exaudflib/vm/swig_vm.h"
#include <jni.h>

#ifdef ENABLE_JAVA_VM

class JavaVMTest;

namespace SWIGVMContainers {

class JavaVMImpl {
    public:
        friend class ::JavaVMTest;
        JavaVMImpl(bool checkOnly, bool noJNI);
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
        void throwException(const char *message);
        void throwException(const std::exception& ex);
        void throwException(const std::string& ex);
        //void throwException(swig_undefined_single_call_exception& ex);
        void importScripts();
        void addExternalJarPaths();
        void getExternalJvmOptions();
        void getScriptClassName();
        void setJvmOptions();
        void addJarToClasspath(const std::string& path);
        std::vector<unsigned char> scriptToMd5(const char *script);
        bool m_checkOnly;
        std::string m_exaJavaPath;
        std::string m_localClasspath;
        std::string m_scriptCode;
        std::string m_exaJarPath;
        std::string m_classpath;
        std::set<std::vector<unsigned char> > m_importedScriptChecksums;
        std::set<std::string> m_jarPaths;
        bool m_exceptionThrown;
        std::vector<std::string> m_jvmOptions;
        JavaVM *m_jvm;
        JNIEnv *m_env;
        bool m_needsCompilation;
};

} //namespace SWIGVMContainers


#endif //ENABLE_JAVA_VM


#endif //JAVACONTAINER_IMPL_H