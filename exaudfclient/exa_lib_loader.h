#pragma once
#include <string>

void* exa_load_libary(const std::string& stdLibPath);
void* exa_load_symbol(void *handle, const std::string& symbol_name);