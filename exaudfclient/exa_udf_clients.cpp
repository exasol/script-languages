#include <iostream>

#include "exa_udf_clients.h"
#include "exa_vm_factory.h"

void ExaUdfClient::usage(const std::string& programName) {
    std::cerr << "Usage: " << programName
              << " <socket> lang=python|lang=java|lang=streaming|lang=benchmark <scriptOptionsParserVersion=1|2>"
              << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClient::create_vm(
    const std::string& languageArg,
    bool useCtpgParser) {
    return ::create_vm(languageArg, useCtpgParser);
}