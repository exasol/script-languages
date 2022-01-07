#include <string>
#include <vector>
#include <set>

#include <jni.h>
#include "exaudflib/exaudflib.h"
#include "exascript_java_jni_decl.h"

class JavaVMTest;

class SWIGVMContainers::JavaVMImpl {
    public:
        friend class ::JavaVMTest;
        JavaVMImpl(bool checkOnly, bool noJNI);
        ~JavaVMImpl() {}
        void shutdown();
        bool run();
        const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall);
    private:
        void createJvm();
        void addPackageToScript();
        void compileScript();
        bool check(const std::string& errorCode, std::string& calledUndefinedSingleCall); // returns 0 if the check failed
        void registerFunctions();
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
        vector<unsigned char> scriptToMd5(const char *script);
        bool m_checkOnly;
        string m_exaJavaPath;
        string m_localClasspath;
        string m_scriptCode;
        string m_exaJarPath;
        string m_classpath;
        set<string> m_jarPaths;
        std::set< std::vector<unsigned char> > m_importedScriptChecksums;
        bool m_exceptionThrown;
        std::vector<std::string> m_jvmOptions;
        JavaVM *m_jvm;
        JNIEnv *m_env;
};
