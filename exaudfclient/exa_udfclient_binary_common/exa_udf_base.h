#pragma once
#include <functional>
#include <string>

#include "exaudflib/vm/swig_vm.h"

typedef bool (*SET_SWIGVM_PARAMS)(SWIGVMContainers::SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVMContainers::SWIGVM*()> vmMaker, int, char**);

class ExaUdfClientBase {
public:
    virtual ~ExaUdfClientBase() = default;
    virtual void usage(const std::string& programName) = 0;
    virtual std::function<SWIGVMContainers::SWIGVM*()> create_vm() = 0;
    virtual bool validate_arguments(int argc, char** argv) = 0;
    virtual void parse_arguments(int argc, char** argv) = 0;
    virtual int startClientBase(int argc, char** argv);
};