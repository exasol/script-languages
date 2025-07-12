#ifndef UDF_PLUGIN_CLIENT
#include <dlfcn.h> //This is required for dynamic linking in new linker namespace, not required for plugins
#endif

#include <sstream>
#include "base/exaudflib/vm/swig_vm.h"

void* handle;

#ifndef UDF_PLUGIN_CLIENT
void* load_dynamic(const char* name) {
    void* res = dlsym(handle, name);
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

