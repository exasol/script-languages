#include <cstdlib>
#include <cstdio>
#include <clocale>
#include <cstring>
#include <cerrno>

#include "exa_set_env.h"

void setup_environment() {
    if (::setenv("HOME", "/tmp", 1) == -1) {
        fprintf(stderr, "Failed setting HOME env var: %s\n", std::strerror(errno));
    }
    ::setlocale(LC_ALL, "en_US.utf8");
}