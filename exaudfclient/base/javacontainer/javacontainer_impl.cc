#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <set>
#include <jni.h>
#include <unistd.h>
#include <sstream>

#include "exascript_java_jni_decl.h"

#include "utils/debug_message.h"
#include "javacontainer/javacontainer.h"
#include "javacontainer/javacontainer_impl.h"
#include "javacontainer/script_options/extractor.h"


using namespace SWIGVMContainers;
using namespace std;

JavaVMImpl::JavaVMImpl(bool checkOnly, bool noJNI,
                        std::unique_ptr<JavaScriptOptions::Extractor> extractor)
: m_checkOnly(checkOnly)
, m_exaJavaPath("")
, m_localClasspath("/tmp") // **IMPORTANT**: /tmp needs to be in the classpath, otherwise ExaCompiler crashe with com.exasol.ExaCompilationException: /DATE_STRING.java:3: error: error while writing DATE_STRING: could not create parent directories
, m_scriptCode(SWIGVM_params->script_code)
, m_jvm(NULL)
, m_env(NULL)
, m_needsCompilation(true)
{

    stringstream ss;
    m_exaJavaPath = "/exaudf/external/exaudfclient_base+/javacontainer"; // TODO hardcoded path

    parseScriptOptions(std::move(extractor));

    m_needsCompilation = checkNeedsCompilation();
    if (m_needsCompilation) {
        DBG_FUNC_CALL(cerr,addPackageToScript());
        DBG_FUNC_CALL(cerr,addLocalClasspath());
    }
    DBG_FUNC_CALL(cerr,setJvmOptions());
    if(false == noJNI) {
        DBG_FUNC_CALL(cerr,createJvm());
        DBG_FUNC_CALL(cerr,registerFunctions());
        if (m_needsCompilation) {
            DBG_FUNC_CALL(cerr,compileScript());
        }
    }
}

void JavaVMImpl::parseScriptOptions(std::unique_ptr<JavaScriptOptions::Extractor> extractor) {

    DBG_FUNC_CALL(cerr,extractor->extract(m_scriptCode));

    DBG_FUNC_CALL(cerr,setClasspath());

    m_jvmOptions = std::move(extractor->moveJvmOptions());

    extractor->iterateJarPaths([&](const std::string& s) { addJarToClasspath(s);});
}

void JavaVMImpl::shutdown() {
    if (m_checkOnly)
        throw JavaVMach::exception("F-UDF.CL.SL.JAVA-1159: Java VM in check only mode");
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    string calledUndefinedSingleCall;
    check("F-UDF.CL.SL.JAVA-1160",calledUndefinedSingleCall);
    if (!cls)
        throw JavaVMach::exception("F-UDF.CL.SL.JAVA-1161: FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "cleanup", "()V");
    check("F-UDF.CL.SL.JAVA-1162",calledUndefinedSingleCall);
    if (!mid)
        throw JavaVMach::exception("F-UDF.CL.SL.JAVA-1163: GetStaticMethodID for run failed");
    m_env->CallStaticVoidMethod(cls, mid);
    check("F-UDF.CL.SL.JAVA-1164",calledUndefinedSingleCall);
    try {
        m_jvm->DestroyJavaVM();
    } catch(...) { 
    
    }
}

bool JavaVMImpl::run() {
    if (m_checkOnly)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1008: Java VM in check only mode");
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    string calledUndefinedSingleCall;
    check("F-UDF-CL-SL-JAVA-1009",calledUndefinedSingleCall);
    if (!cls)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1010: FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "run", "()V");
    check("F-UDF-CL-SL-JAVA-1011",calledUndefinedSingleCall);
    if (!mid)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1012: GetStaticMethodID for run failed");
    m_env->CallStaticVoidMethod(cls, mid);
    check("F-UDF-CL-SL-JAVA-1013",calledUndefinedSingleCall);
    return true;
}

static string singleCallResult;

const char* JavaVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall) {

    if (m_checkOnly)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1014: Java VM in check only mode");

    const char* func = NULL;
    switch (fn) {
        case SC_FN_NIL: break;
        case SC_FN_DEFAULT_OUTPUT_COLUMNS: func = "getDefaultOutputColumns"; break;
        case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL: func = "adapterCall"; break;
        case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC: func = "generateSqlForImportSpec"; break;
        case SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC: func = "generateSqlForExportSpec"; break;
    }
    if (func == NULL) {
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1015: Unknown single call "+std::to_string(fn));
    }
    jclass cls = m_env->FindClass("com/exasol/ExaWrapper");
    check("F-UDF-CL-SL-JAVA-1016",calledUndefinedSingleCall);
    if (!cls)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1017: FindClass for ExaWrapper failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "runSingleCall", "(Ljava/lang/String;Ljava/lang/Object;)[B");
    check("F-UDF-CL-SL-JAVA-1018",calledUndefinedSingleCall);
    if (!mid)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1019: GetStaticMethodID for run failed");
    jstring fn_js = m_env->NewStringUTF(func);
    check("F-UDF-CL-SL-JAVA-1020",calledUndefinedSingleCall);


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
            check("F-UDF-CL-SL-JAVA-1021",calledUndefinedSingleCall);
            jmethodID import_spec_wrapper_constructor = m_env->GetMethodID(import_spec_wrapper_cls, "<init>", "(JZ)V");
            check("F-UDF-CL-SL-JAVA-1022",calledUndefinedSingleCall);
            args_js = m_env->NewObject(import_spec_wrapper_cls, import_spec_wrapper_constructor, &imp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC) {
        exp_spec = const_cast<ExecutionGraph::ExportSpecification*>(dynamic_cast<const ExecutionGraph::ExportSpecification*>(&args));
        exp_spec_wrapper = ExecutionGraph::ExportSpecificationWrapper(exp_spec);
        if (exp_spec)
        {
            jclass export_spec_wrapper_cls = m_env->FindClass("com/exasol/swig/ExportSpecificationWrapper");
            check("F-UDF-CL-SL-JAVA-1023",calledUndefinedSingleCall);
            jmethodID export_spec_wrapper_constructor = m_env->GetMethodID(export_spec_wrapper_cls, "<init>", "(JZ)V");
            check("F-UDF-CL-SL-JAVA-1024",calledUndefinedSingleCall);
            args_js = m_env->NewObject(export_spec_wrapper_cls, export_spec_wrapper_constructor, &exp_spec_wrapper, false);
        }
    } else if (fn == SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL) {
        const ExecutionGraph::StringDTO* argDto = dynamic_cast<const ExecutionGraph::StringDTO*>(&args);
        string string_arg = argDto->getArg();
        args_js = m_env->NewStringUTF(string_arg.c_str());
    }
    
    check("F-UDF-CL-SL-JAVA-1025",calledUndefinedSingleCall);
    jbyteArray resJ = (jbyteArray)m_env->CallStaticObjectMethod(cls, mid, fn_js, args_js);
    if (check("F-UDF-CL-SL-JAVA-1026",calledUndefinedSingleCall) == 0) return strdup("<error during singleCall>");
    jsize resLen = m_env->GetArrayLength(resJ);
    check("F-UDF-CL-SL-JAVA-1027",calledUndefinedSingleCall);
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
        ss << "F-UDF-CL-SL-JAVA-1028: Cannot start the JVM: ";
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
        throw JavaVMach::exception(ss.str());
    }
    delete [] options;
}

void JavaVMImpl::compileScript() {
    string calledUndefinedSingleCall;
    jstring classnameStr = m_env->NewStringUTF(SWIGVM_params->script_name);
    check("F-UDF-CL-SL-JAVA-1029",calledUndefinedSingleCall);
    jstring codeStr = m_env->NewStringUTF(m_scriptCode.c_str());
    check("F-UDF-CL-SL-JAVA-1030",calledUndefinedSingleCall);
    jstring classpathStr = m_env->NewStringUTF(m_localClasspath.c_str());
    check("F-UDF-CL-SL-JAVA-1031",calledUndefinedSingleCall);
    if (!classnameStr || !codeStr || !classpathStr)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1032: NewStringUTF for compile failed");
    jclass cls = m_env->FindClass("com/exasol/ExaCompiler");
    check("F-UDF-CL-SL-JAVA-1033",calledUndefinedSingleCall);
    if (!cls)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1034: FindClass for ExaCompiler failed");
    jmethodID mid = m_env->GetStaticMethodID(cls, "compile", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
    check("F-UDF-CL-SL-JAVA-1035",calledUndefinedSingleCall);
    if (!mid)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1036: GetStaticMethodID for compile failed");
    m_env->CallStaticVoidMethod(cls, mid, classnameStr, codeStr, classpathStr);
    check("F-UDF-CL-SL-JAVA-1037",calledUndefinedSingleCall);
}


bool JavaVMImpl::check(const string& errorCode, string& calledUndefinedSingleCall) {
    jthrowable ex = m_env->ExceptionOccurred();
    if (ex) {
        m_env->ExceptionClear();

        jclass undefinedSingleCallExceptionClass = m_env->FindClass("com/exasol/ExaUndefinedSingleCallException");
        if (!undefinedSingleCallExceptionClass) {
            throw JavaVMach::exception(errorCode+": F-UDF-CL-SL-JAVA-1042: FindClass for com.exasol.ExaUndefinedSingleCallException failed");
        }
        if (m_env->IsInstanceOf(ex, undefinedSingleCallExceptionClass)) {
            jmethodID undefinedRemoteFn = m_env->GetMethodID(undefinedSingleCallExceptionClass, "getUndefinedRemoteFn", "()Ljava/lang/String;");
            check("F-UDF-CL-SL-JAVA-1043",calledUndefinedSingleCall);
            if (!undefinedRemoteFn)
                throw JavaVMach::exception(errorCode+": F-UDF-CL-SL-JAVA-1044: com.exasol.ExaUndefinedSingleCallException.getUndefinedRemoteFn() could not be found");
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
               throw JavaVMach::exception(errorCode+": F-UDF-CL-SL-JAVA-1045: Internal error: getUndefinedRemoteFn() returned no result");
            } 
        }

        string exceptionMessage = "";
        jclass exClass = m_env->GetObjectClass(ex);
        if (!exClass)
            throw JavaVMach::exception(errorCode+": F-UDF-CL-SL-JAVA-1046: FindClass for Throwable failed");
        // Throwable.toString()
        jmethodID toString = m_env->GetMethodID(exClass, "toString", "()Ljava/lang/String;");
        check("F-UDF-CL-SL-JAVA-1047",calledUndefinedSingleCall);
        if (!toString)
            throw JavaVMach::exception(errorCode+": F-UDF-CL-SL-JAVA-1048: Throwable.toString() could not be found");
        jobject object = m_env->CallObjectMethod(ex, toString);
        if (object) {
            jstring message = static_cast<jstring>(object);
            char const *utfMessage = m_env->GetStringUTFChars(message, 0);
            exceptionMessage.append("\n");
            exceptionMessage.append(utfMessage);
            m_env->ReleaseStringUTFChars(message, utfMessage);
        }
        else {
            exceptionMessage.append(errorCode+": F-UDF-CL-SL-JAVA-1049: Throwable.toString(): result is null");
        }

//        // Build Stacktrace
//
//        // Throwable.getStackTrace()
//        jmethodID getStackTrace = m_env->GetMethodID(exClass, "getStackTrace", "()[Ljava/lang/StackTraceElement;");
//        check("F-UDF-CL-SL-JAVA-1050",calledUndefinedSingleCall);
//        if (!getStackTrace)
//            throwException(errorCode+": F-UDF-CL-SL-JAVA-1051: Throwable.getStackTrace() could not be found");
//        jobjectArray frames = (jobjectArray)m_env->CallObjectMethod(ex, getStackTrace);
//        if (frames) {
//            jclass frameClass = m_env->FindClass("java/lang/StackTraceElement");
//            check("F-UDF-CL-SL-JAVA-1052",calledUndefinedSingleCall);
//            if (!frameClass)
//                throwException(errorCode+": F-UDF-CL-SL-JAVA-1053: FindClass for StackTraceElement failed");
//            jmethodID frameToString = m_env->GetMethodID(frameClass, "toString", "()Ljava/lang/String;");
//            check("F-UDF-CL-SL-JAVA-1054",calledUndefinedSingleCall);
//            if (!frameToString)
//                throwException(errorCode+": F-UDF-CL-SL-JAVA-1055: StackTraceElement.toString() could not be found");
//            jsize framesLength = m_env->GetArrayLength(frames);
//            for (int i = 0; i < framesLength; i++) {
//                jobject frame = m_env->GetObjectArrayElement(frames, i);
//                check("F-UDF-CL-SL-JAVA-1056",calledUndefinedSingleCall);
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
        throw JavaVMach::exception(errorCode+": "+exceptionMessage);
    }
    return 1;
}

void JavaVMImpl::registerFunctions() {
    string calledUndefinedSingleCall;
    jclass cls = m_env->FindClass("com/exasol/swig/exascript_javaJNI");
    check("F-UDF-CL-SL-JAVA-1057",calledUndefinedSingleCall);
    if (!cls)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1058: FindClass for exascript_javaJNI failed");
    int rc = m_env->RegisterNatives(cls, methods, sizeof(methods) / sizeof(methods[0]));
    check("F-UDF-CL-SL-JAVA-1059",calledUndefinedSingleCall);
    if (rc)
        throw JavaVMach::exception("F-UDF-CL-SL-JAVA-1060: RegisterNatives failed");
}

void JavaVMImpl::setClasspath() {
    m_exaJarPath = m_exaJavaPath + "/exaudf_deploy.jar";
    m_classpath = m_exaJarPath;
}

void JavaVMImpl::addLocalClasspath() {
    m_classpath = m_localClasspath + ":" + m_classpath;
}

bool JavaVMImpl::checkNeedsCompilation() {
    std::string trimmedScriptCode = m_scriptCode;
    trimmedScriptCode.erase(0, trimmedScriptCode.find_first_not_of("\t\n\r ")); // left trim
    trimmedScriptCode.erase(trimmedScriptCode.find_last_not_of("\t\n\r ") + 1); // right trim
    return false == trimmedScriptCode.empty();
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
            errorMsg << "F-UDF-CL-SL-JAVA-1061: Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
            throw JavaVMach::exception(errorMsg.str());
        }
    }
    else if (rc) {
        stringstream errorMsg;
        errorMsg << "F-UDF-CL-SL-JAVA-1062: Java VM cannot find '" << jarPath.c_str() << "': " << strerror(errno);
        throw JavaVMach::exception(errorMsg.str());
    }

    if (!S_ISREG(st.st_mode)) {
        stringstream errorMsg;
        errorMsg << "F-UDF-CL-SL-JAVA-1063: '" << jarPath.c_str() << "' is not a regular file";
        throw JavaVMach::exception(errorMsg.str());
    }

    // Add file to classpath
    m_classpath += ":" + jarPath;
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
