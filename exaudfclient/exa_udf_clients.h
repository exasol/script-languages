#pragma once
#include <string>

#include "exa_udf_base.h"
#include "vm/swig_vm.h"

class ExaUdfClientPython : public ExaUdfClientBase {
public:
    ~ExaUdfClientPython() override = default;

    void usage(const std::string& programName) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm() override;
};

class ExaUdfClientJava : public ExaUdfClientBase {
public:
    ~ExaUdfClientJava() override = default;

    void usage(const std::string& programName) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm() override;
};

class ExaUdfStreaming : public ExaUdfClientBase {
public:
    ~ExaUdfStreaming() override = default;

    void usage(const std::string& programName) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm() override;
};

class ExaUdfClientBenchmark : public ExaUdfClientBase {
public:
    ~ExaUdfClientBenchmark() override = default;

    void usage(const std::string& programName) override;
    std::function<SWIGVMContainers::SWIGVM*()> create_vm() override;
};