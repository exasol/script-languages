#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <clocale>
#include <cstring>
#include <cerrno>

#include "exa_set_env.h"

void setup_environment() {
    if (::setenv("HOME", "/tmp", 1) == -1) {
        std::cerr << "Failed setting HOME env var: " << std::strerror(errno) << std::endl;
    }
    if (::setlocale(LC_ALL, "en_US.utf8") == nullptr) {
        std::cerr << "Failed setting locale: " << std::strerror(errno) << std::endl;
    }
}