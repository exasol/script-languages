#include "base/javacontainer/javacontainer.h"
#include "base/javacontainer/javacontainer_impl.h"
#include "base/javacontainer/script_options/extractor.h"

using namespace SWIGVMContainers;
using namespace std;

JavaVMach::JavaVMach(bool checkOnly,std::unique_ptr<JavaScriptOptions::Extractor> extractor) {
    try {
        m_impl = new JavaVMImpl(checkOnly, false, std::move(extractor));
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1000: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1001: JVM crashed for unknown reason";
    }
}


bool JavaVMach::run() {
    try {
        return m_impl->run();
    }  catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1002: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1003: JVM crashed for unknown reason";
    }
    return false;
}

void JavaVMach::shutdown() {
    try {
        m_impl->shutdown();
    }  catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1004: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1005: JVM crashed for unknown reason";
    }
}

const char* JavaVMach::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn, args, calledUndefinedSingleCall);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1006: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-JAVA-1007: JVM crashed for unknown reason";
    }
    return strdup("<this is an error>");
}
