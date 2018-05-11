#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <openssl/md5.h>
#include <set>
#include <jni.h>
#include <swigcontainers_ext.h>
#include <exascript_java_jni_decl.h>
#include <unistd.h>

#include "scriptoptionlines.h"

using namespace SWIGVMContainers;
using namespace std;

class SWIGVMContainers::JavaVMImpl {
    public:
        JavaVMImpl(bool checkOnly);
        ~JavaVMImpl() {}
        void shutdown();
        bool run();
        std::string singleCall(single_call_function_id fn, const ExecutionGraph::ScriptDTO& args);
    private:
        void createJvm();
        void addPackageToScript();
        void compileScript();
        void check();
        void registerFunctions();
        void setClasspath();
        void throwException(const char *message);
        void throwException(std::exception& ex);
        void throwException(swig_undefined_single_call_exception& ex);
        void importScripts();
        void addExternalJarPaths();
        void getExternalJvmOptions();
        void getScriptClassName();
        void setJvmOptions();
        void addJarToClasspath(const string& path);
        vector<unsigned char> scriptToMd5(const char *script);
        bool m_checkOnly;
        string m_exaJavaPath;
        string m_localClasspath;
        string m_scriptCode;
        string m_exaJarPath;
        string m_classpath;
        set<string> m_jarPaths;
        set< vector<unsigned char> > m_importedScriptChecksums;
        bool m_exceptionThrown;
        vector<string> m_jvmOptions;
        JavaVM *m_jvm;
        JNIEnv *m_env;
};

JavaVMach::JavaVMach(bool checkOnly): m_impl(new JavaVMImpl(checkOnly)) {
}


bool JavaVMach::run() {
    return m_impl->run();
}

void JavaVMach::shutdown() {m_impl->shutdown();}

std::string JavaVMach::singleCall(single_call_function_id fn, const ExecutionGraph::ScriptDTO& args) {
    return m_impl->singleCall(fn, args);
}

JavaVMImpl::JavaVMImpl(bool checkOnly): m_checkOnly(checkOnly), m_exaJavaPath(""), m_localClasspath("/tmp"),
                                        m_scriptCode(SWIGVM_params->script_code), m_exceptionThrown(false), m_jvm(NULL), m_env(NULL) {

    stringstream ss;

    m_exaJavaPath = "/exasol";

    setClasspath();
    getScriptClassName();  // To be called before scripts are imported. Otherwise, the script classname from an imported script could be used
    importScripts();
    addPackageToScript();
    addExternalJarPaths();
    getExternalJvmOptions();
    setJvmOptions();
    createJvm();
    registerFunctions();
    compileScript();
}

void JavaVMImpl::shutdown() {
    try {
        m_jvm->DestroyJavaVM();
    } catch(...) { }
}

bool JavaVMImpl::run() {
    if (m_checkOnly)
        throwException("Java VM in check only mode");
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    check();
    if (!cls)
        throwException("FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "run", "()V");
    check();
    if (!mid)
        throwException("GetStaticMethodID for run failed");
    m_env->CallStaticVoidMethod(cls, mid);
    check();
    return true;
}

std::string JavaVMImpl::singleCall(single_call_function_id fn, const ExecutionGraph::ScriptDTO& args) {
    if (m_checkOnly)
        throwException("Java VM in check only mode");

    const char* func = NULL;
    switch (fn) {
    case SC_FN_NIL: break;
    case SC_FN_DEFAULT_OUTPUT_COLUMNS: func = "getDefaultOutputColumns"; break;
    case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL: func = "adapterCall"; break;
    case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC: func = "generateSqlForImportSpec"; break;
    case SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC: func = "generateSqlForExportSpec"; break;
    }
    if (func == NULL)
        abort();
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    check();
    if (!cls)
        throwException("FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "runSingleCall", "(Ljava/lang/String;Ljava/lang/Object;)[B");
    check();
    if (!mid)
        throwException("GetStaticMethodID for run failed");
    jstring fn_js = m_env->NewStringUTF(func);
    check();

    // Prepare arg
    // TODO VS This will be refactored completely
    // We intentionally define these variables outside so that they are not destroyed too early
    jobject args_js = NULL;
    ExecutionGraph::ImportSpecification* imp_spec;
    ExecutionGraph::ImportSpecificationWrapper imp_spec_wrapper(NULL);
    ExecutionGraph::ExportSpecification* exp_spec;
    ExecutionGraph::ExportSpecificationWrapper exp_spec_wrapper(NULL);
    if (fn == SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC) {
        imp_spec = const_cast<ExecutionGraph::ImportSpecification*>(dynamic_cast<const ExecutionGraph::ImportSpecification*>(&args));
        imp_spec_wrapper = ExecutionGraph::ImportSpecificationWrapper(imp_spec);
        if (imp_spec)
        {         
            jclass import_spec_wrapper_cls = m_env->FindClass("com/exasol/swig/ImportSpecificationWrapper");
            check();
            jmethodID import_spec_wrapper_constructor = m_env->GetMethodID(import_spec_wrapper_cls, "<init>", "(JZ)V");
            check();
            args_js = m_env->NewObject(import_spec_wrapper_cls, import_spec_wrapper_constructor, &imp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC) {
        exp_spec = const_cast<ExecutionGraph::ExportSpecification*>(dynamic_cast<const ExecutionGraph::ExportSpecification*>(&args));
        exp_spec_wrapper = ExecutionGraph::ExportSpecificationWrapper(exp_spec);
        if (exp_spec)
        {
            jclass export_spec_wrapper_cls = m_env->FindClass("com/exasol/swig/ExportSpecificationWrapper");
            check();
            jmethodID export_spec_wrapper_constructor = m_env->GetMethodID(export_spec_wrapper_cls, "<init>", "(JZ)V");
            check();
            args_js = m_env->NewObject(export_spec_wrapper_cls, export_spec_wrapper_constructor, &exp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL) {
        const ExecutionGraph::StringDTO* argDto = dynamic_cast<const ExecutionGraph::StringDTO*>(&args);
        string string_arg = argDto->getArg();
        args_js = m_env->NewStringUTF(string_arg.c_str());
    }
    
    check();
    jbyteArray resJ = (jbyteArray)m_env->CallStaticObjectMethod(cls, mid, fn_js, args_js);
    check();
    jsize resLen = m_env->GetArrayLength(resJ);
    check();
    char* buffer = new char[resLen + 1];
    m_env->GetByteArrayRegion(resJ, 0, resLen, reinterpret_cast<jbyte*>(buffer));
    buffer[resLen] = '\0';
    string res = string(buffer);
    delete buffer;
    return res;
}

void JavaVMImpl::addPackageToScript() {
    // Each script is generated in the com.exasol package, not in the default
    // package. Scripts classes may not be in the default package because
    // com.exasol.ExaWrapper requires to call run() on them (accessing default
    // package from a non-default package is not allowed). Furthermore, only if
    // the script is in the same package as ExaWrapper the script can be
    // defined as package-private.
    m_scriptCode = "package com.exasol;\r\n" + m_scriptCode;
}

void JavaVMImpl::createJvm() {
    unsigned int numJvmOptions = m_jvmOptions.size();
    JavaVMOption *options = new JavaVMOption[numJvmOptions];
    for (size_t i = 0; i < numJvmOptions; ++i) {
        options[i].optionString = (char*)(m_jvmOptions[i].c_str());
        options[i].extraInfo = NULL;
    }

    JavaVMInitArgs vm_args;
    vm_args.version = JNI_VERSION_1_6;
    vm_args.nOptions = numJvmOptions;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_FALSE;

    int rc = JNI_CreateJavaVM(&m_jvm, (void**)&m_env, &vm_args);
    if (rc != JNI_OK) {
        stringstream ss;
        ss << "Cannot start the JVM: ";
        switch (rc) {
            case JNI_ERR: ss << "unknown error"; break;
            case JNI_EDETACHED: ss << "thread is detached from VM"; break;
            case JNI_EVERSION: ss << "version error"; break;
            case JNI_ENOMEM: ss << "out of memory"; break;
            case JNI_EEXIST: ss << "VM already exists"; break;
            case JNI_EINVAL: ss << "invalid arguments"; break;
            default: ss << "unknown"; break;
        }
        ss << " (" << rc << ")";
        delete [] options;
        throwException(ss.str().c_str());
    }
    delete [] options;
}

void JavaVMImpl::compileScript() {
    jstring classnameStr = m_env->NewStringUTF(SWIGVM_params->script_name);
    check();
    jstring codeStr = m_env->NewStringUTF(m_scriptCode.c_str());
    check();
    jstring classpathStr = m_env->NewStringUTF(m_localClasspath.c_str());
    check();
    if (!classnameStr || !codeStr || !classpathStr)
        throwException("NewStringUTF for compile failed");
    jclass cls = m_env->FindClass("com/exasol/ExaCompiler");
    check();
    if (!cls)
        throwException("FindClass for ExaCompiler failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "compile", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
    check();
    if (!mid)
        throwException("GetStaticMethodID for compile failed");
    m_env->CallStaticVoidMethod(cls, mid, classnameStr, codeStr, classpathStr);
    check();
}

void JavaVMImpl::addExternalJarPaths() {
    const string jarKeyword = "%jar";
    const string whitespace = " \t\f\v";
    const string lineEnd = ";";
    size_t pos;
    while (true) {
        string jarPath = ExecutionGraph::extractOptionLine(m_scriptCode, jarKeyword, whitespace, lineEnd, pos, [&](const char* msg){throwException(msg);});
        if (jarPath == "")
            break;
        for (size_t start = 0, delim = 0; ; start = delim + 1) {
            delim = jarPath.find(":", start);
            if (delim != string::npos) {
                string jar = jarPath.substr(start, delim - start);
                if (m_jarPaths.find(jar) == m_jarPaths.end())
                    m_jarPaths.insert(jar);
            }
            else {
                string jar = jarPath.substr(start);
                if (m_jarPaths.find(jar) == m_jarPaths.end())
                    m_jarPaths.insert(jar);
                break;
            }
        }
    }
    for (set<string>::iterator it = m_jarPaths.begin(); it != m_jarPaths.end(); ++it) {
        addJarToClasspath(*it);
    }
}

void JavaVMImpl::getScriptClassName() {
    const string scriptClassKeyword = "%scriptclass";
    const string whitespace = " \t\f\v";
    const string lineEnd = ";";
    size_t pos;
    string scriptClass = ExecutionGraph::extractOptionLine(m_scriptCode, scriptClassKeyword, whitespace, lineEnd, pos, [&](const char* msg){throwException(msg);});
    if (scriptClass != "") {
        m_jvmOptions.push_back("-Dexasol.scriptclass=" + scriptClass);
    }
}

void JavaVMImpl::importScripts() {
    SWIGMetadata *meta = NULL;
    const string importKeyword = "%import";
    const string whitespace = " \t\f\v";
    const string lineEnd = ";";
    size_t pos;
    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports itself
    m_importedScriptChecksums.insert(scriptToMd5(m_scriptCode.c_str()));
    while (true) {
        string scriptName = ExecutionGraph::extractOptionLine(m_scriptCode, importKeyword, whitespace, lineEnd, pos, [&](const char* msg){throwException(msg);});
        if (scriptName == "")
            break;
        if (!meta) {
            meta = new SWIGMetadata();
            if (!meta)
                throwException("Failure while importing scripts");
        }
        const char *scriptCode = meta->moduleContent(scriptName.c_str());
        const char *exception = meta->checkException();
        if (exception)
            throwException(exception);
        if (m_importedScriptChecksums.insert(scriptToMd5(scriptCode)).second) {
            // Script has not been imported yet
            // If this imported script contains %import statements 
            // they will be resolved in this while loop.
            m_scriptCode.insert(pos, scriptCode);
        }
    }
    if (meta)
        delete meta;
}

void JavaVMImpl::check() {
    jthrowable ex = m_env->ExceptionOccurred();
    if (ex) {
        m_env->ExceptionClear();

        jclass undefinedSingleCallExceptionClass = m_env->FindClass("com/exasol/ExaUndefinedSingleCallException");
        if (!undefinedSingleCallExceptionClass) {
            throwException("FindClass for com.exasol.ExaUndefinedSingleCallException failed");
        }
        if (m_env->IsInstanceOf(ex, undefinedSingleCallExceptionClass)) {
            jmethodID undefinedRemoteFn = m_env->GetMethodID(undefinedSingleCallExceptionClass, "getUndefinedRemoteFn", "()Ljava/lang/String;");
            check();
            if (!undefinedRemoteFn)
                throwException("com.exasol.ExaUndefinedSingleCallException.getUndefinedRemoteFn() could not be found");
            jobject undefinedRemoteFnString = m_env->CallObjectMethod(ex,undefinedRemoteFn);
            if (undefinedRemoteFnString) {
                jstring fn = static_cast<jstring>(undefinedRemoteFnString);
                const char *fn_str = m_env->GetStringUTFChars(fn,0);
                std::string fn_string = fn_str;
                m_env->ReleaseStringUTFChars(fn,fn_str);  
                swig_undefined_single_call_exception ex(fn_string);
                throwException(ex);
            } else {
               throwException("Internal error: getUndefinedRemoteFn() returned no result"); 
            } 
        }

        string exceptionMessage = "";
        jclass exClass = m_env->GetObjectClass(ex);
        if (!exClass)
            throwException("FindClass for Throwable failed");
        // Throwable.toString()
        jmethodID toString = m_env->GetMethodID(exClass, "toString", "()Ljava/lang/String;");
        check();
        if (!toString)
            throwException("Throwable.toString() could not be found");
        jobject object = m_env->CallObjectMethod(ex, toString);
        if (object) {
            jstring message = static_cast<jstring>(object);
            char const *utfMessage = m_env->GetStringUTFChars(message, 0);
            exceptionMessage.append("\n");
            exceptionMessage.append(utfMessage);
            m_env->ReleaseStringUTFChars(message, utfMessage);
        }
        else {
            exceptionMessage.append("Throwable.toString(): result is null");
        }
        // Throwable.getStackTrace()
        jmethodID getStackTrace = m_env->GetMethodID(exClass, "getStackTrace", "()[Ljava/lang/StackTraceElement;");
        check();
        if (!getStackTrace)
            throwException("Throwable.getStackTrace() could not be found");
        jobjectArray frames = (jobjectArray)m_env->CallObjectMethod(ex, getStackTrace);
        if (frames) {
            jclass frameClass = m_env->FindClass("java/lang/StackTraceElement");
            check();
            if (!frameClass)
                throwException("FindClass for StackTraceElement failed");
            jmethodID frameToString = m_env->GetMethodID(frameClass, "toString", "()Ljava/lang/String;");
            check();
            if (!frameToString)
                throwException("StackTraceElement.toString() could not be found");
            jsize framesLength = m_env->GetArrayLength(frames);
            for (int i = 0; i < framesLength; i++) {
                jobject frame = m_env->GetObjectArrayElement(frames, i);
                check();
                jobject frameMsgObj = m_env->CallObjectMethod(frame, frameToString);
                if (frameMsgObj) {
                    jstring message = static_cast<jstring>(frameMsgObj);
                    const char *utfMessage = m_env->GetStringUTFChars(message, 0);
                    string line = utfMessage;
                    // If stack trace lines are wrapper or reflection entries, stop
                    if ((line.find("ExaWrapper.") == 0) || (line.find("ExaCompiler.") == 0) || (line.find("sun.reflect.") == 0)) {
                        if (i != 0)
                            exceptionMessage.append("\n");
                        m_env->ReleaseStringUTFChars(message, utfMessage);
                        break;
                    }
                    else {
                        if (i == 0)
                            exceptionMessage.append("\nStack trace:");
                        exceptionMessage.append("\n");
                        exceptionMessage.append(utfMessage);
                        m_env->ReleaseStringUTFChars(message, utfMessage);
                    }
                }
            }
        }
        throwException(exceptionMessage.c_str());
    }
}

void JavaVMImpl::registerFunctions() {
    jclass cls = m_env->FindClass("com/exasol/swig/exascript_javaJNI");
    check();
    if (!cls)
        throwException("FindClass for exascript_javaJNI failed");
    int rc = m_env->RegisterNatives(cls, methods, sizeof(methods) / sizeof(methods[0]));
    check();
    if (rc)
        throwException("RegisterNatives failed");
}

void JavaVMImpl::setClasspath() {
    m_exaJarPath = m_exaJavaPath + "/exaudf.jar";
    m_classpath = m_localClasspath + ":" + m_exaJarPath;
}

vector<unsigned char> JavaVMImpl::scriptToMd5(const char *script) {
    MD5_CTX ctx;
    unsigned char md5[MD5_DIGEST_LENGTH];
    MD5_Init(&ctx);
    MD5_Update(&ctx, script, strlen(script));
    MD5_Final(md5, &ctx);
    return vector<unsigned char>(md5, md5 + sizeof(md5));
}

void JavaVMImpl::addJarToClasspath(const string& path) {
    string jarPath = path; // m_exaJavaPath + "/jars/" + path;

    struct stat st;
    int rc = stat(jarPath.c_str(), &st);
    if (rc && errno == ENOENT) {
        // Not found in HOME, try path directly
        jarPath = path;
        rc = stat(jarPath.c_str(), &st);
        if (rc) {
            stringstream errorMsg;
            errorMsg << "Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
            throwException(errorMsg.str().c_str());
        }
    }
    else if (rc) {
        stringstream errorMsg;
        errorMsg << "Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
        throwException(errorMsg.str().c_str());
    }

    if (!S_ISREG(st.st_mode)) {
        stringstream errorMsg;
        errorMsg << "'" << jarPath.c_str() << "' is not a regular file";
        throwException(errorMsg.str().c_str());
    }

    // Add file to classpath
    m_classpath += ":" + jarPath;
}

void JavaVMImpl::getExternalJvmOptions() {
    const string jvmOption = "%jvmoption";
    const string whitespace = " \t\f\v";
    const string lineEnd = ";";
    size_t pos;
    while (true) {
        string options = ExecutionGraph::extractOptionLine(m_scriptCode,jvmOption, whitespace, lineEnd, pos, [&](const char* msg){throwException(msg);});
        if (options == "")
            break;
        for (size_t start = 0, delim = 0; ; start = delim + 1) {
            start = options.find_first_not_of(whitespace, start);
            if (start == string::npos)
                break;
            delim = options.find_first_of(whitespace, start);
            if (delim != string::npos) {
                m_jvmOptions.push_back(options.substr(start, delim - start));
            }
            else {
                m_jvmOptions.push_back(options.substr(start));
                break;
            }
        }
    }
}

void JavaVMImpl::setJvmOptions() {
    bool minHeap = false;
    bool maxHeap = false;
    bool maxStack = false;
    for (vector<string>::iterator it = m_jvmOptions.begin(); it != m_jvmOptions.end(); ++it) {
        if ((*it).find("-Xms") != string::npos) {
            minHeap = true;
        }
        else if ((*it).find("-Xmx") != string::npos) {
            maxHeap = true;
        }
        else if ((*it).find("-Xss") != string::npos) {
            maxStack = true;
        }
    }

    // Initial heap size
    unsigned long minHeapSize = 128;
    unsigned long maxHeapSize = 1024;
    if (!minHeap) {
        stringstream ss;
        ss << "-Xms" << minHeapSize << "m";
        m_jvmOptions.push_back(ss.str());
    }
    // Max heap size
    if (!maxHeap) {
        unsigned long maxHeapSizeMb = static_cast<long>(0.625 * (SWIGVM_params->maximal_memory_limit / (1024 * 1024)));
        maxHeapSizeMb = (maxHeapSizeMb < minHeapSize) ? minHeapSize : maxHeapSizeMb;
        maxHeapSizeMb = (maxHeapSizeMb > maxHeapSize) ? maxHeapSize : maxHeapSizeMb;
        stringstream ss;
        ss << "-Xmx" << maxHeapSizeMb << "m";
        m_jvmOptions.push_back(ss.str());
    }
    // Max thread stack size
    if (!maxStack)
        m_jvmOptions.push_back("-Xss512k");
    // Error file path
    m_jvmOptions.push_back("-XX:ErrorFile=" + m_localClasspath + "/hs_err_pid%p.log");
    // Classpath
    m_jvmOptions.push_back("-Djava.class.path=" + m_classpath);
    // Serial garbage collection
    m_jvmOptions.push_back("-XX:+UseSerialGC");
}



void JavaVMImpl::throwException(const char *message) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw JavaVMach::exception(message);
}

void JavaVMImpl::throwException(std::exception& ex) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw ex;
}

void JavaVMImpl::throwException(swig_undefined_single_call_exception& ex) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw ex;
}
