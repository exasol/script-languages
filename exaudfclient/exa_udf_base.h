#pragma once
#include <functional>
#include <string>

#include "vm/swig_vm.h"

typedef bool (*SET_SWIGVM_PARAMS)(SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVMContainers::SWIGVM*()> vmMaker, int, char**);

class ExaUdfClientBase {
public:
    virtual ~ExaUdfClientBase() = default;
    virtual void usage(const std::string& programName) = 0;
    virtual std::function<SWIGVMContainers::SWIGVM*()> create_vm(
        const std::string& languageArg,
        bool useCtpgParser) = 0;

    bool validate_arguments(int argc, char** argv);
    void parse_arguments(int argc, char** argv);
    int startClientBase(int argc, char** argv);
    
protected:
    std::string m_socket;
    std::string m_languageArg;
    bool mb_useCtpgParser = false;
};