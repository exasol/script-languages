#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <fstream>
#include <link.h>
#include <string.h>

#ifndef UDF_PLUGIN_CLIENT
#include <dlfcn.h> //This is required for dynamic linking in new linker namespace, not required for plugins
#endif
#include <exception>
#include "base/exaudflib/vm/swig_vm.h"
#include "base/exaudflib/load_dynamic.h"
#ifdef ENABLE_BENCHMARK_VM
#include "benchmark_container/benchmark_container.h"
#endif
#ifdef ENABLE_STREAMING_VM
#include "streaming_container/streamingcontainer.h"
#endif
#include <functional>
#include "base/utils/debug_message.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#ifdef __cplusplus
# define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>



#ifdef ENABLE_JAVA_VM
#include "base/javacontainer/javacontainer_builder.h"
#endif //ENABLE_JAVA_VM

#ifdef ENABLE_PYTHON_VM
#include "base/python/pythoncontainer.h"
#endif //ENABLE_PYTHON_VM

#ifdef UDF_PLUGIN_CLIENT
#include "protegrityclient.h"
#endif

using namespace std;
using namespace SWIGVMContainers;

namespace SWIGVMContainers {
__thread SWIGVM_params_t * SWIGVM_params = nullptr;
}

typedef bool (*VOID_FUN_WITH_SWIGVM_PARAMS_P)(SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVM*()>vmMaker,int,char**);

#ifdef UDF_PLUGIN_CLIENT
extern "C" {
int exaudfclient_main(std::function<SWIGVM*()>vmMaker,int argc,char**argv);
void set_SWIGVM_params(SWIGVM_params_t* p);
}
#endif

int main(int argc, char **argv) {
   const std::string origLdLibraryPath(::getenv("LD_LIBRARY_PATH"));
   const std::string newLdLibraryPath = std::string("/opt/conda/cuda-compat/:") + origLdLibraryPath;
   ::setenv("LD_LIBRARY_PATH", newLdLibraryPath.c_str());
#ifndef UDF_PLUGIN_CLIENT
#ifdef CUSTOM_LIBEXAUDFLIB_PATH
    std::string libexaudflibPath = string(CUSTOM_LIBEXAUDFLIB_PATH);
#else
    std::string libexaudflibPath = ::getenv("LIBEXAUDFLIB_PATH");
    //std::string libexaudflibPath="libexaudflib_complete.so";
    //std::string libexaudflibPath = std::string(argv[3]);
    //std::string libexaudflibPath = std::string("/exaudf/libexaudflib_complete.so");
#endif
#if 1

    Lmid_t  my_namespace_id;
    // DBGMSG(std::cerr, "Load libprotobuf into new namespace");
    // DBGVAR(std::cerr, libProtobufPath);
    // handle = dlmopen(LM_ID_NEWLM, libProtobufPath.c_str(),RTLD_NOW);
    // if (!handle) {
    //     std::cerr << "Error when dynamically loading libprotobuf: " << dlerror() << endl;
    //     exit(EXIT_FAILURE);
    // }
    // if(dlinfo(handle, RTLD_DI_LMID, &my_namespace_id) != 0) {
    //     cerr << "Error when getting namespace id " << dlerror() << endl;
    //     exit(EXIT_FAILURE);
    // }
    DBGMSG(cerr, "Load libexaudflib");
    DBGVAR(cerr, libexaudflibPath);
    handle = dlmopen(LM_ID_NEWLM, libexaudflibPath.c_str(), RTLD_NOW);
//    handle = dlopen(libexaudflibPath.c_str(), RTLD_NOW);

    if (!handle) {
        fprintf(stderr, "dmlopen: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
#else
    handle = dlopen(libProtobufPath.c_str(),RTLD_NOW|RTLD_GLOBAL);
    if (!handle) {
        cerr << "Error when dynamically loading libprotobuf: " << dlerror() << endl;
        exit(EXIT_FAILURE);
    }
    handle = dlopen("/exaudf/libexaudflib.so",RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
#endif


    MAIN_FUN exaudfclient_main = (MAIN_FUN)load_dynamic("exaudfclient_main");
    VOID_FUN_WITH_SWIGVM_PARAMS_P set_SWIGVM_params = (VOID_FUN_WITH_SWIGVM_PARAMS_P)load_dynamic("set_SWIGVM_params");


#endif  // ifndef UDF_PLUGIN_CLIENT

#ifdef UDF_PLUGIN_CLIENT
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <socket>" << endl;
        return 1;
    }
#else
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <socket> lang=python|lang=r|lang=java|lang=streaming|lang=benchmark" << endl;
        return 1;
    }
    const char* script_options_parser_env_val = ::getenv("SCRIPT_OPTIONS_PARSER_VERSION");
    const bool useCtpgScriptOptionsParser = script_options_parser_env_val != nullptr &&
                                            ::strcmp(script_options_parser_env_val, "2") == 0;
#endif

    if (::setenv("HOME", "/tmp", 1) == -1)
    {
        throw SWIGVM::exception("Failed to set HOME directory");
    }
    ::setlocale(LC_ALL, "en_US.utf8");

    std::function<SWIGVMContainers::SWIGVM*()>vmMaker=[](){return nullptr;}; // the initial vm maker returns NULL
#ifdef UDF_PLUGIN_CLIENT
    vmMaker = [](){return new SWIGVMContainers::Protegrity(false);};
#else
    if (strcmp(argv[2], "lang=python")==0)
    {
#ifdef ENABLE_PYTHON_VM
        char *path_var = getenv("PATH");
        if (path_var != NULL) {
            std::string path_var_str = std::string(path_var);
            path_var_str.insert(0, "/opt/conda/bin:");
            if (::setenv("PATH", path_var_str.c_str(), 1) == -1) {
                cerr << "Unable to prefix PATH env variable with /opt/conda/bin";
            }
        }

        vmMaker = [](){return new  SWIGVMContainers::PythonVM(false);};
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without Python support");
#endif
    } else if (strcmp(argv[2], "lang=java")==0)
    {
#ifdef ENABLE_JAVA_VM
        if (useCtpgScriptOptionsParser) {
                vmMaker = [&](){return SWIGVMContainers::JavaContainerBuilder().useCtpgParser().build();};
        } else {
            vmMaker = [&](){return SWIGVMContainers::JavaContainerBuilder().build();};
        }
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without Java support");
#endif
    } else if (strcmp(argv[2], "lang=streaming")==0)
    {
#ifdef ENABLE_STREAMING_VM
            vmMaker = [](){return new StreamingVM(false);};
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without Streaming support");
#endif
    } else if (strcmp(argv[2], "lang=benchmark")==0){
#ifdef ENABLE_BENCHMARK_VM
        vmMaker = [](){return new BenchmarkVM(false);};
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without Benchmark support");
#endif
    }
#endif

    SWIGVM_params = new SWIGVM_params_t(true);
    set_SWIGVM_params(SWIGVM_params);



    return exaudfclient_main(vmMaker, argc, argv);
    
}
