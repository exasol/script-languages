#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <clocale>
#include <cstring>
#include <cerrno>

#include "exaudflib/vm/swig_vm.h"
#include "exa_set_env.h"

namespace exa_env {
    void setup_environment() {
        if (::setenv("HOME", "/tmp", 1) == -1) {
            std::cerr << "Failed setting HOME env var: " << std::strerror(errno) << std::endl;
            throw SWIGVMContainers::SWIGVM::exception("Failed to set HOME directory");
        }
        if (::setlocale(LC_ALL, "en_US.utf8") == nullptr) {
            std::cerr << "Failed setting locale: " << std::strerror(errno) << std::endl;
            throw SWIGVMContainers::SWIGVM::exception("Failed to set locale");
        }
    }
}