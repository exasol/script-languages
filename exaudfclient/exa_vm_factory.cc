#include <functional>
#include "exa_vm_factory.h"

std::function<SWIGVMContainers::SWIGVM*()> create_vm(const std::string& argv_lang) {
#ifdef UDF_PLUGIN_CLIENT
    return [](){return new SWIGVMContainers::Protegrity(false);};
#else
    if(argv_lang.compare("lang=python") == 0) {
        #ifdef ENABLE_PYTHON_VM
            return []() { return new  SWIGVMContainers::PythonVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Python support");
        #endif
    }
    else if(argv_lang.compare("lang=java") == 0) {
        #ifdef ENABLE_JAVA_VM
            return []() { return SWIGVMContainers::JavaContainerBuilder().build(); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Java support");
        #endif
    }
    else if(argv_lang.compare("lang=streaming") == 0) {
        #ifdef ENABLE_STREAMING_VM
            return []() { return new StreamingVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Streaming support");
        #endif
    }
    else if(argv_lang.compare("lang=benchmark") == 0) {
        #ifdef ENABLE_BENCHMARK_VM
            return []() { return new BenchmarkVM(false); };
        #else
            throw SWIGVMContainers::SWIGVM::exception("this exaudfclient has been compilied without Benchmark support");
        #endif
    }
    else {
        throw SWIGVMContainers::SWIGVM::exception("unsupported language specified in argv");
    }   
#endif
}