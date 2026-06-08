#pragma once
#include <string>

#include "exa_udf_base.h"
#include "vm/swig_vm.h"

class ExaUdfClient : public ExaUdfClientBase {
public:
    ~ExaUdfClient() override = default;

    void usage(const std::string& programName) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm(
        const std::string& languageArg,
        bool useCtpgParser) override;
};