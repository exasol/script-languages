#pragma once
#include <string>

namespace exa_lib {
    void* load_library(const std::string& stdLibPath);
    void* load_symbol(void *handle, const std::string& symbol_name);
}