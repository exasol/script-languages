#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <openssl/md5.h>
#include <set>
#include <jni.h>
#include "exaudflib/exaudflib.h"
#include "exascript_java_jni_decl.h"
#include <unistd.h>

#include "debug_message.h"
#include "exaudflib/scriptoptionlines.h"

using namespace SWIGVMContainers;
using namespace std;

class SWIGVMContainers::JavaVMImpl {
    public:
        JavaVMImpl(bool checkOnly);
        ~JavaVMImpl() {}
        void shutdown();
        bool run();
        const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall);
    private:
        void createJvm();
        void addPackageToScript();
        void compileScript();
        bool check(const string& errorCode, string& calledUndefinedSingleCall); // returns 0 if the check failed
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

JavaVMach::JavaVMach(bool checkOnly) {
    try {
        m_impl = new JavaVMImpl(checkOnly);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-1: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-2: some unknown exception occurred";
    }
}


bool JavaVMach::run() {
    try {
        return m_impl->run();
    }  catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-3: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-4: some unknown exception occurred";
    }
    return false;
}

void JavaVMach::shutdown() {
    try {
        m_impl->shutdown();
    }  catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-5: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-6: some unknown exception occurred";
    }
}

const char* JavaVMach::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn, args, calledUndefinedSingleCall);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-7: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF.CL.J-8: some unknown exception occurred";
    }
    return strdup("<this is an error>");
}

JavaVMImpl::JavaVMImpl(bool checkOnly): m_checkOnly(checkOnly), m_exaJavaPath(""), m_localClasspath("/tmp"), // **IMPORTANT**: /tmp needs to be in the classpath, otherwise ExaCompiler crashe with com.exasol.ExaCompilationException: /DATE_STRING.java:3: error: error while writing DATE_STRING: could not create parent directories
                                        m_scriptCode(SWIGVM_params->script_code), m_exceptionThrown(false), m_jvm(NULL), m_env(NULL) {

    stringstream ss;
    m_exaJavaPath = "/exaudf/javacontainer"; // TODO hardcoded path
    DBG_FUNC_CALL(cerr,setClasspath());
    DBG_FUNC_CALL(cerr,getScriptClassName());  // To be called before scripts are imported. Otherwise, the script classname from an imported script could be used
    DBG_FUNC_CALL(cerr,importScripts());
    DBG_FUNC_CALL(cerr,addPackageToScript());
    DBG_FUNC_CALL(cerr,addExternalJarPaths());
    DBG_FUNC_CALL(cerr,getExternalJvmOptions());
    DBG_FUNC_CALL(cerr,setJvmOptions());
    DBG_FUNC_CALL(cerr,createJvm());
    DBG_FUNC_CALL(cerr,registerFunctions());
    DBG_FUNC_CALL(cerr,compileScript());
}

void JavaVMImpl::shutdown() {
    try {
        m_jvm->DestroyJavaVM();
    } catch(...) { 
    
    }
}

bool JavaVMImpl::run() {
    if (m_checkOnly)
        throwException("F-UDF.CL.J-9: Java VM in check only mode");
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    string calledUndefinedSingleCall;
    check("F-UDF.CL.J-133",calledUndefinedSingleCall);
    if (!cls)
        throwException("F-UDF.CL.J-10: FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "run", "()V");
    check("F-UDF.CL.J-134",calledUndefinedSingleCall);
    if (!mid)
        throwException("F-UDF.CL.J-11: GetStaticMethodID for run failed");
    m_env->CallStaticVoidMethod(cls, mid);
    check("F-UDF.CL.J-135",calledUndefinedSingleCall);
    return true;
}

static string singleCallResult;

const char* JavaVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall) {

    if (m_checkOnly)
        throwException("F-UDF.CL.J-12: Java VM in check only mode");

    const char* func = NULL;
    switch (fn) {
        case SC_FN_NIL: break;
        case SC_FN_DEFAULT_OUTPUT_COLUMNS: func = "getDefaultOutputColumns"; break;
        case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL: func = "adapterCall"; break;
        case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC: func = "generateSqlForImportSpec"; break;
        case SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC: func = "generateSqlForExportSpec"; break;
    }
    if (func == NULL) {
        throwException("F-UDF.CL.J-13: Unknown single call "+std::to_string(fn));
    }
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    check("F-UDF.CL.J-136",calledUndefinedSingleCall);
    if (!cls)
        throwException("F-UDF.CL.J-14: FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "runSingleCall", "(Ljava/lang/String;Ljava/lang/Object;)[B");
    check("F-UDF.CL.J-137",calledUndefinedSingleCall);
    if (!mid)
        throwException("F-UDF.CL.J-15: GetStaticMethodID for run failed");
    jstring fn_js = m_env->NewStringUTF(func);
    check("F-UDF.CL.J-138",calledUndefinedSingleCall);


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
            check("F-UDF.CL.J-139",calledUndefinedSingleCall);
            jmethodID import_spec_wrapper_constructor = m_env->GetMethodID(import_spec_wrapper_cls, "<init>", "(JZ)V");
            check("F-UDF.CL.J-140",calledUndefinedSingleCall);
            args_js = m_env->NewObject(import_spec_wrapper_cls, import_spec_wrapper_constructor, &imp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC) {
        exp_spec = const_cast<ExecutionGraph::ExportSpecification*>(dynamic_cast<const ExecutionGraph::ExportSpecification*>(&args));
        exp_spec_wrapper = ExecutionGraph::ExportSpecificationWrapper(exp_spec);
        if (exp_spec)
        {
            jclass export_spec_wrapper_cls = m_env->FindClass("com/exasol/swig/ExportSpecificationWrapper");
            check("F-UDF.CL.J-141",calledUndefinedSingleCall);
            jmethodID export_spec_wrapper_constructor = m_env->GetMethodID(export_spec_wrapper_cls, "<init>", "(JZ)V");
            check("F-UDF.CL.J-142",calledUndefinedSingleCall);
            args_js = m_env->NewObject(export_spec_wrapper_cls, export_spec_wrapper_constructor, &exp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL) {
        const ExecutionGraph::StringDTO* argDto = dynamic_cast<const ExecutionGraph::StringDTO*>(&args);
        string string_arg = argDto->getArg();
        args_js = m_env->NewStringUTF(string_arg.c_str());
    }
    
    check("F-UDF.CL.J-143",calledUndefinedSingleCall);
    jbyteArray resJ = (jbyteArray)m_env->CallStaticObjectMethod(cls, mid, fn_js, args_js);
    if (check("F-UDF.CL.J-144",calledUndefinedSingleCall) == 0) return strdup("<error during singleCall>");
    jsize resLen = m_env->GetArrayLength(resJ);
    check("F-UDF.CL.J-145",calledUndefinedSingleCall);
    char* buffer = new char[resLen + 1];
    m_env->GetByteArrayRegion(resJ, 0, resLen, reinterpret_cast<jbyte*>(buffer));
    buffer[resLen] = '\0';
    singleCallResult = string(buffer);
    delete buffer;
    m_env->DeleteLocalRef(args_js);
    m_env->DeleteLocalRef(resJ);


    return singleCallResult.c_str();
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
        options[i].optionString = strdup((char*)(m_jvmOptions[i].c_str()));
        DBGVAR(cerr,options[i].optionString);
        options[i].extraInfo = NULL;
    }

    JavaVMInitArgs vm_args;
    vm_args.version = JNI_VERSION_1_2;
    vm_args.nOptions = numJvmOptions;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_FALSE;
    //vm_args.ignoreUnrecognized = JNI_TRUE;
    DBGVAR(cerr,m_env);
    
    DBG_FUNC_CALL(cerr,int rc = JNI_CreateJavaVM(&m_jvm, (void**)&m_env, &vm_args));
    if (rc != JNI_OK) {
        stringstream ss;
        ss << "F-UDF.CL.J-16: Cannot start the JVM: ";
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
    string calledUndefinedSingleCall;
    jstring classnameStr = m_env->NewStringUTF(SWIGVM_params->script_name);
    check("F-UDF.CL.J-146",calledUndefinedSingleCall);
    jstring codeStr = m_env->NewStringUTF(m_scriptCode.c_str());
    check("F-UDF.CL.J-147",calledUndefinedSingleCall);
    jstring classpathStr = m_env->NewStringUTF(m_localClasspath.c_str());
    check("F-UDF.CL.J-148",calledUndefinedSingleCall);
    if (!classnameStr || !codeStr || !classpathStr)
        throwException("F-UDF.CL.J-17: NewStringUTF for compile failed");
    jclass cls = m_env->FindClass("com/exasol/ExaCompiler");
    check("F-UDF.CL.J-149",calledUndefinedSingleCall);
    if (!cls)
        throwException("F-UDF.CL.J-18: FindClass for ExaCompiler failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "compile", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
    check("F-UDF.CL.J-150",calledUndefinedSingleCall);
    if (!mid)
        throwException("F-UDF.CL.J-19: GetStaticMethodID for compile failed");
    m_env->CallStaticVoidMethod(cls, mid, classnameStr, codeStr, classpathStr);
    check("F-UDF.CL.J-151",calledUndefinedSingleCall);
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
    string scriptClass = 
      ExecutionGraph::extractOptionLine(
          m_scriptCode, 
          scriptClassKeyword, 
          whitespace, 
          lineEnd, 
          pos, 
          [&](const char* msg){throwException("F-UDF.CL.J-20: "+std::string(msg));}
          );
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
    // package definition). Otherwise we don't recognize if the script imports its self
    m_importedScriptChecksums.insert(scriptToMd5(m_scriptCode.c_str()));
    while (true) {
        string scriptName = 
          ExecutionGraph::extractOptionLine(
              m_scriptCode, 
              importKeyword, 
              whitespace, 
              lineEnd, 
              pos, 
              [&](const char* msg){throwException("F-UDF.CL.J-21: "+std::string(msg));}
              );
        if (scriptName == "")
            break;
        if (!meta) {
            meta = new SWIGMetadata();
            if (!meta)
                throwException("F-UDF.CL.J-22: Failure while importing scripts");
        }
        const char *scriptCode = meta->moduleContent(scriptName.c_str());
        const char *exception = meta->checkException();
        if (exception)
            throwException("F-UDF.CL.J-23: "+std::string(exception));
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

bool JavaVMImpl::check(const string& errorCode, string& calledUndefinedSingleCall) {
    jthrowable ex = m_env->ExceptionOccurred();
    if (ex) {
        m_env->ExceptionClear();

        jclass undefinedSingleCallExceptionClass = m_env->FindClass("com/exasol/ExaUndefinedSingleCallException");
        if (!undefinedSingleCallExceptionClass) {
            throwException(errorCode+": F-UDF.CL.J-38: FindClass for com.exasol.ExaUndefinedSingleCallException failed");
        }
        if (m_env->IsInstanceOf(ex, undefinedSingleCallExceptionClass)) {
            jmethodID undefinedRemoteFn = m_env->GetMethodID(undefinedSingleCallExceptionClass, "getUndefinedRemoteFn", "()Ljava/lang/String;");
            check("F-UDF.CL.J-152",calledUndefinedSingleCall);
            if (!undefinedRemoteFn)
                throwException(errorCode+": F-UDF.CL.J-24: com.exasol.ExaUndefinedSingleCallException.getUndefinedRemoteFn() could not be found");
            jobject undefinedRemoteFnString = m_env->CallObjectMethod(ex,undefinedRemoteFn);
            if (undefinedRemoteFnString) {
                jstring fn = static_cast<jstring>(undefinedRemoteFnString);
                const char *fn_str = m_env->GetStringUTFChars(fn,0);
                std::string fn_string = fn_str;
                m_env->ReleaseStringUTFChars(fn,fn_str); 
                calledUndefinedSingleCall = fn_string; 
                return 0;
                //swig_undefined_single_call_exception ex(fn_string);
                //throwException(ex);
            } else {
               throwException(errorCode+": F-UDF.CL.J-25: Internal error: getUndefinedRemoteFn() returned no result"); 
            } 
        }

        string exceptionMessage = "";
        jclass exClass = m_env->GetObjectClass(ex);
        if (!exClass)
            throwException(errorCode+": F-UDF.CL.J-26: FindClass for Throwable failed");
        // Throwable.toString()
        jmethodID toString = m_env->GetMethodID(exClass, "toString", "()Ljava/lang/String;");
        check("F-UDF.CL.J-153",calledUndefinedSingleCall);
        if (!toString)
            throwException(errorCode+": F-UDF.CL.J-27: Throwable.toString() could not be found");
        jobject object = m_env->CallObjectMethod(ex, toString);
        if (object) {
            jstring message = static_cast<jstring>(object);
            char const *utfMessage = m_env->GetStringUTFChars(message, 0);
            exceptionMessage.append("\n");
            exceptionMessage.append(utfMessage);
            m_env->ReleaseStringUTFChars(message, utfMessage);
        }
        else {
            exceptionMessage.append(errorCode+": F-UDF.CL.J-28: Throwable.toString(): result is null");
        }

//        // Build Stacktrace
//
//        // Throwable.getStackTrace()
//        jmethodID getStackTrace = m_env->GetMethodID(exClass, "getStackTrace", "()[Ljava/lang/StackTraceElement;");
//        check("F-UDF.CL.J-154",calledUndefinedSingleCall);
//        if (!getStackTrace)
//            throwException(errorCode+": F-UDF.CL.J-29: Throwable.getStackTrace() could not be found");
//        jobjectArray frames = (jobjectArray)m_env->CallObjectMethod(ex, getStackTrace);
//        if (frames) {
//            jclass frameClass = m_env->FindClass("java/lang/StackTraceElement");
//            check("F-UDF.CL.J-155",calledUndefinedSingleCall);
//            if (!frameClass)
//                throwException(errorCode+": F-UDF.CL.J-30: FindClass for StackTraceElement failed");
//            jmethodID frameToString = m_env->GetMethodID(frameClass, "toString", "()Ljava/lang/String;");
//            check("F-UDF.CL.J-156",calledUndefinedSingleCall);
//            if (!frameToString)
//                throwException(errorCode+": F-UDF.CL.J-31: StackTraceElement.toString() could not be found");
//            jsize framesLength = m_env->GetArrayLength(frames);
//            for (int i = 0; i < framesLength; i++) {
//                jobject frame = m_env->GetObjectArrayElement(frames, i);
//                check("F-UDF.CL.J-157",calledUndefinedSingleCall);
//                jobject frameMsgObj = m_env->CallObjectMethod(frame, frameToString);
//                if (frameMsgObj) {
//                    jstring message = static_cast<jstring>(frameMsgObj);
//                    const char *utfMessage = m_env->GetStringUTFChars(message, 0);
//                    string line = utfMessage;
//                    // If stack trace lines are wrapper or reflection entries, stop
//                    if ((line.find("ExaWrapper.") == 0) || (line.find("ExaCompiler.") == 0) || (line.find("sun.reflect.") == 0)) {
//                        if (i != 0)
//                            exceptionMessage.append("\n");
//                        m_env->ReleaseStringUTFChars(message, utfMessage);
//                        break;
//                    }
//                    else {
//                        if (i == 0)
//                            exceptionMessage.append("\nStack trace:");
//                        exceptionMessage.append("\n");
//                        exceptionMessage.append(utfMessage);
//                        m_env->ReleaseStringUTFChars(message, utfMessage);
//                    }
//                }
//            }
//        }
        throwException(errorCode+": "+exceptionMessage);
    }
    return 1;
}

void JavaVMImpl::registerFunctions() {
    string calledUndefinedSingleCall;
    jclass cls = m_env->FindClass("com/exasol/swig/exascript_javaJNI");
    check("F-UDF.CL.J-158",calledUndefinedSingleCall);
    if (!cls)
        throwException("F-UDF.CL.J-32: FindClass for exascript_javaJNI failed");
    int rc = m_env->RegisterNatives(cls, methods, sizeof(methods) / sizeof(methods[0]));
    check("F-UDF.CL.J-159",calledUndefinedSingleCall);
    if (rc)
        throwException("F-UDF.CL.J-33: RegisterNatives failed");
}

void JavaVMImpl::setClasspath() {
    m_exaJarPath = m_exaJavaPath + "/libexaudf.jar";
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
            errorMsg << "F-UDF.CL.J-34: Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
            throwException(errorMsg.str().c_str());
        }
    }
    else if (rc) {
        stringstream errorMsg;
        errorMsg << "F-UDF.CL.J-35: Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
        throwException(errorMsg.str().c_str());
    }

    if (!S_ISREG(st.st_mode)) {
        stringstream errorMsg;
        errorMsg << "F-UDF.CL.J-36: '" << jarPath.c_str() << "' is not a regular file";
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
        string options = 
          ExecutionGraph::extractOptionLine(
              m_scriptCode,
              jvmOption, 
              whitespace, 
              lineEnd, 
              pos, 
              [&](const char* msg){throwException("F-UDF.CL.J-37: "+std::string(msg));}
              );
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
    m_jvmOptions.push_back("-XX:+UseSerialGC"); // TODO allow different Garbage Collectors, multiple options are not allowed, so we need to check if options was specified by the user or otherwise use -XX:+UseSerialGC as default
}


void JavaVMImpl::throwException(const std::string&  message) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw JavaVMach::exception(message);
}


void JavaVMImpl::throwException(const char*  message) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw JavaVMach::exception(message);
}

void JavaVMImpl::throwException(const std::exception& ex) {
    if (!m_exceptionThrown) {
        m_exceptionThrown = true;
    }
    throw ex;
}

//void JavaVMImpl::throwException(swig_undefined_single_call_exception& ex) {
//    if (!m_exceptionThrown) {
//        m_exceptionThrown = true;
//    }
//    throw ex;
//}
