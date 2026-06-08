#include <functional>
#include "exa_vm_factory.h"

#ifdef ENABLE_JAVA_VM
#include "javacontainer/javacontainer_builder.h"
#endif //ENABLE_JAVA_VM

#ifdef ENABLE_PYTHON_VM
#include "python/pythoncontainer.h"
#endif //ENABLE_PYTHON_VM

#ifdef ENABLE_STREAMING_VM
#include "streaming_container/streamingcontainer.h"
#endif

#ifdef ENABLE_BENCHMARK_VM
#include "benchmark_container/benchmark_container.h"
#endif

std::function<SWIGVMContainers::SWIGVM*()> create_vm(const std::string& argv_lang, bool use_ctpg_options_parser) {
    if(argv_lang.compare("lang=python") == 0) {
        #ifdef ENABLE_PYTHON_VM
            return []() { return new  SWIGVMContainers::PythonVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Python support");
        #endif
    }
    else if(argv_lang.compare("lang=java") == 0) {
        #ifdef ENABLE_JAVA_VM
            if (use_ctpg_options_parser) {
                return [&](){return SWIGVMContainers::JavaContainerBuilder().useCtpgParser().build();};
            } else {
                return [&](){return SWIGVMContainers::JavaContainerBuilder().build();};
            }
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Java support");
        #endif
    }
    else if(argv_lang.compare("lang=streaming") == 0) {
        #ifdef ENABLE_STREAMING_VM
            return []() { return new SWIGVMContainers::StreamingVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Streaming support");
        #endif
    }
    else if(argv_lang.compare("lang=benchmark") == 0) {
        #ifdef ENABLE_BENCHMARK_VM
            return []() { return new SWIGVMContainers::BenchmarkVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Benchmark support");
        #endif
    }
    else {
        throw SWIGVMContainers::SWIGVM::exception("unsupported language specified in argv");
    }   
}