#ifndef UDF_PLUGIN_CLIENT
#include <dlfcn.h> //This is required for dynamic linking in new linker namespace, not required for plugins
#endif

#include <sstream>
#include "exaudflib/vm/swig_vm.h"

static void* exaudflib_handle = nullptr;

#ifndef UDF_PLUGIN_CLIENT

void set_exaudflib_handle(void* handle) {
    if(handle != nullptr) {
        exaudflib_handle = handle;
    }
}

void* load_dynamic(const char* name) {
    if(exaudflib_handle == nullptr) {
        throw SWIGVMContainers::SWIGVM::exception("Error: exaudflib_handle is not set.");
    }
    void* res = dlsym(exaudflib_handle, name);
    char* error;
    if ((error = dlerror()) != nullptr)
    {
        std::stringstream sb;
        sb << "Error when trying to load function '" << name << "': " << error;
        throw SWIGVMContainers::SWIGVM::exception(sb.str().c_str());
    }
    return res;
}
#endif

