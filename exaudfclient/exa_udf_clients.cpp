#include "exa_udf_python.h"
#include "vm/swig_vm.h"


//----------------------------------------Python UDF Client----------------------------------------
#ifdef ENABLE_PYTHON_VM
#include "python/pythoncontainer.h"
#endif //ENABLE_PYTHON_VM

void ExaUdfClientPython::usage(const std::string& programName) {
    std::cerr   << "Usage: " << programName << " <socket> lang=python <scriptOptionsParserVersion=1|2>"
                << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClientPython::create_vm() {
    #ifdef ENABLE_PYTHON_VM
        return []() { return new  SWIGVMContainers::PythonVM(false); };
    #else
        throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Python support");
    #endif
}


//----------------------------------------Java UDF Client----------------------------------------
#ifdef ENABLE_JAVA_VM
#include "javacontainer/javacontainer_builder.h"
#endif //ENABLE_JAVA_VM

void ExaUdfClientJava::usage(const std::string& programName) {
    std::cerr   << "Usage: " << programName << " <socket> lang=java <scriptOptionsParserVersion=1|2>"
                << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClientJava::create_vm() {
    #ifdef ENABLE_JAVA_VM
        if (mb_useCtpgParser) {
            return [&](){return SWIGVMContainers::JavaContainerBuilder().useCtpgParser().build();};
        } else {
            return [&](){return SWIGVMContainers::JavaContainerBuilder().build();};
        }
    #else
        throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Java support");
    #endif
}


//----------------------------------------Streaming UDF Client----------------------------------------
#ifdef ENABLE_STREAMING_VM
#include "streaming_container/streamingcontainer.h"
#endif

void ExaUdfClientStreaming::usage(const std::string& programName) {
    std::cerr   << "Usage: " << programName << " <socket> lang=streaming <scriptOptionsParserVersion=1|2>"
                << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClientStreaming::create_vm() {
    #ifdef ENABLE_STREAMING_VM
        return []() { return new  SWIGVMContainers::StreamingVM(false); };
    #else
        throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Streaming support");
    #endif
}


//----------------------------------------Benchmark UDF Client----------------------------------------
#ifdef ENABLE_BENCHMARK_VM
#include "benchmark_container/benchmark_container.h"
#endif

void ExaUdfClientBenchmark::usage(const std::string& programName) {
    std::cerr   << "Usage: " << programName << " <socket> lang=benchmark <scriptOptionsParserVersion=1|2>"
                << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClientBenchmark::create_vm() {
    #ifdef ENABLE_BENCHMARK_VM
        return []() { return new  SWIGVMContainers::BenchmarkVM(false); };
    #else
        throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Benchmark support");
    #endif
}