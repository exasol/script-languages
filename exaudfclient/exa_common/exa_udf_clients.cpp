#include <iostream>
#include <cstring>

#include "exa_udf_clients.h"
#include "exa_vm_factory.h"

void ExaUdfClient::parse_arguments(int argc, char** argv) {
    //  Now assumption is that all cmd line aguments are already validated.
    m_languageArg = argv[2];
   
    m_useCtpgParser = false;
    const char* env_val = ::getenv("SCRIPT_OPTIONS_PARSER_VERSION");
    if(env_val && strcmp(env_val, "2") == 0) {
        m_useCtpgParser = true;
    } else if(argc == 4) {
        std::string parse_option(argv[3]);
        m_useCtpgParser =  (parse_option.compare("scriptOptionsParserVersion=2") == 0);
    }
}

bool ExaUdfClient::validate_arguments(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return false;
    }

    if (argc == 4 && 
        strcmp(argv[3], "scriptOptionsParserVersion=1") != 0 &&
        strcmp(argv[3], "scriptOptionsParserVersion=2") != 0) {
        usage(argv[0]);
        return false;
    }

    if (!((strcmp(argv[2], "lang=python") == 0)
        || (strcmp(argv[2], "lang=java") == 0)
        || (strcmp(argv[2], "lang=streaming") == 0)
        || (strcmp(argv[2], "lang=benchmark") == 0))) {
        usage(argv[0]);
        return false;
    }

    return true;
}

void ExaUdfClient::usage(const std::string& programName) {
    std::cerr << "Usage: " << programName
              << " <socket> lang=python|lang=java|lang=streaming|lang=benchmark <scriptOptionsParserVersion=1|2>"
              << std::endl;
}

std::function<SWIGVMContainers::SWIGVM*()> ExaUdfClient::create_vm() {
    return ::create_vm(m_languageArg, m_useCtpgParser);
}
