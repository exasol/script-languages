#pragma once
#include <string>

#include "exa_udf_base.h"
#include "exaudflib/vm/swig_vm.h"

class ExaUdfClient : public ExaUdfClientBase {
public:
    ~ExaUdfClient() override = default;
    void parse_arguments(int argc, char** argv) override;
    void usage(const std::string& programName) override;
    bool validate_arguments(int argc, char** argv) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm() override;
    
protected:
    std::string m_languageArg;
    bool m_useCtpgParser = false;
};