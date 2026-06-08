#include <iostream>
#include <string>
#include <sstream>
#include <dlfcn.h>
#include <dlfcn.h>

#include "exa_lib_loader.h"
#include "utils/debug_message.h"

void* exa_load_libary(const std::string& stdLibPath) {
    if (stdLibPath.empty()) {
        return nullptr;
    }
    
    void* handle = dlmopen(LM_ID_NEWLM, stdLibPath.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "dlmopen error: " << dlerror() << "; while loading " << stdLibPath << std::endl;
        return nullptr;
    }
    return handle;
}

void* exa_load_symbol(void *handle, const std::string& symbol_name) {
    void *p_res = nullptr;
    char *error = nullptr;
    if(handle) {
        p_res = dlsym(handle, symbol_name.c_str());

        if((error = dlerror()) != nullptr) {
            std::cerr << "Error when trying to load symbol '" << symbol_name << "': " << error << std::endl;
            return nullptr;
        }
    }
    return p_res;
}

