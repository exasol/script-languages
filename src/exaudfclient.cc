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
#ifndef PROTEGRITY_PLUGIN_CLIENT
#include <dlfcn.h>
#endif
#include <exception>
#include "exaudflib.h"
#include "benchmark_container.h"
#include <functional>


#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#ifdef __cplusplus
# define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>


#ifdef PROTEGRITY_PLUGIN_CLIENT
#include "protegrityclient.h"
#endif

using namespace std;
using namespace SWIGVMContainers;

namespace SWIGVMContainers {
__thread SWIGVM_params_t * SWIGVM_params = nullptr;
}

void* handle;

typedef bool (*VOID_FUN_WITH_SWIGVM_PARAMS_P)(SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVM*()>vmMaker,int,char**);

char* error;

#ifndef PROTEGRITY_PLUGIN_CLIENT
void* load_dynamic(const char* name) {
    void* res = dlsym(handle, name);
    if ((error = dlerror()) != NULL)
    {
        std::stringstream sb;
        sb << "Error when trying to load function '" << name << "': " << error;
        throw SWIGVM::exception(sb.str().c_str());
    }
    return res;
}
#endif

#ifdef PROTEGRITY_PLUGIN_CLIENT
extern "C" {
int exaudfclient_main(std::function<SWIGVM*()>vmMaker,int argc,char**argv);
void set_SWIGVM_params(SWIGVM_params_t* p);
}
#endif

int main(int argc, char **argv) {
#ifdef CUSTOM_PROTOBUF_PREFIX
    string libProtobufPath= string(CUSTOM_PROTOBUF_PREFIX) + "/libprotobuf.so";
#else
    string libProtobufPath = "/usr/lib/x86_64-linux-gnu/libprotobuf.so";
#endif
#ifdef CUSTOM_LIBEXAUDFLIB_PATH
    string libexaudflibPath = string(CUSTOM_LIBEXAUDFLIB_PATH);
#else
    string libexaudflibPath = "/exaudf/libexaudflib_complete.so"
#endif
#ifndef PROTEGRITY_PLUGIN_CLIENT
#if 1

    Lmid_t  my_namespace_id;
    cerr << "Load libprotobuf from " << libProtobufPath << endl;
    handle = dlmopen(LM_ID_NEWLM, libProtobufPath.c_str(),RTLD_NOW);
    if (!handle) {
        cerr << "Error when dynamically loading libprotobuf: " << dlerror() << endl;
        exit(EXIT_FAILURE);
    }
    if(dlinfo(handle, RTLD_DI_LMID, &my_namespace_id) != 0) {
        cerr << "Error when getting namespace id " << dlerror() << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "Load libexaudflib from " << libexaudflibPath << endl;
    handle = dlmopen(my_namespace_id, libexaudflibPath.c_str() ,RTLD_NOW);
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


#endif  // ifndef PROTEGRITY_PLUGIN_CLIENT

#ifdef PROTEGRITY_PLUGIN_CLIENT
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <socket>" << endl;
        return 1;
    }
#else
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <socket> lang=python|lang=r|lang=java|lang=streaming|lang=benchmark" << endl;
        return 1;
    }
#endif

    if (::setenv("HOME", "/tmp", 1) == -1)
    {
        throw SWIGVM::exception("Failed to set HOME directory");
    }
    ::setlocale(LC_ALL, "en_US.utf8");

    std::function<SWIGVM*()>vmMaker=[](){return nullptr;}; // the initial vm maker returns NULL

#ifdef PROTEGRITY_PLUGIN_CLIENT
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

        vmMaker = [](){return new PythonVM(false);};
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without Python support");
#endif
    } else if (strcmp(argv[2], "lang=r")==0)
    {
#ifdef ENABLE_R_VM
        vmMaker = [](){return new RVM(false);};
#else
        throw SWIGVM::exception("this exaudfclient has been compilied without R support");
#endif
    } else if (strcmp(argv[2], "lang=java")==0)
    {
#ifdef ENABLE_JAVA_VM
        vmMaker = [](){return new JavaVMach(false);};
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
