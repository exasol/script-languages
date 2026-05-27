#pragma once
#include <functional>

#include "vm/swig_vm.h"

namespace SWIGVMContainers {
__thread SWIGVM_params_t * SWIGVM_params = nullptr;
}

typedef bool (*SET_SWIGVM_PARAMS)(SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVMContainers::SWIGVM*()> vmMaker, int, char**);

enum class ExaUdfLanguage {
    Python3,
    Java,
    Streaming,
    Benchmark
};

class ExaUdfClientBase {
public:
    virtual ~ExaUdfClientBase() = default;
    virtual void usage(const std::string& programName) = 0;
    virtual std::function<SWIGVMContainers::SWIGVM*()> create_vm() = 0;

    bool validate_arguments(int argc, char** argv);
    void parse_arguments(int argc, char** argv);
    int startClientBase(int argc, char** argv);
    bool is_use_ctpg_parser(const std::string& argv_parser_option);

protected:
    std::string m_socket;
    std::string m_lang;
    ExaUdfLanguage m_lang;
    bool mb_useCtpgParser;
};