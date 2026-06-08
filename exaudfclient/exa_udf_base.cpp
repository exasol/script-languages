#include <iostream>
#include <cstring>
#include <cstdlib>
#include <string>

#include "exaudf_lib_output_path.h"
#include "vm/swig_vm.h"
#include "exa_udf_base.h"
#include "exa_lib_loader.h"
#include "exa_set_env.h"
#include "utils/debug_message.h"


void ExaUdfClientBase::parse_arguments(int argc, char** argv) {
    //  Now assumption is that all cmd line aguments are already validated.
    m_socket = argv[1];
    m_languageArg = argv[2];
   
    mb_useCtpgParser = false;
    const char* env_val = ::getenv("SCRIPT_OPTIONS_PARSER_VERSION");
    if(env_val && strcmp(env_val, "2") == 0) {
        mb_useCtpgParser = true;
    } else if(argc == 4) {
        std::string parse_option(argv[3]);
        mb_useCtpgParser =  (parse_option.compare("scriptOptionsParserVersion=2") == 0);
    }
}


bool ExaUdfClientBase::validate_arguments(int argc, char** argv) {
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

int ExaUdfClientBase::startClientBase(int argc, char** argv) {
    if (!validate_arguments(argc, argv)) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    parse_arguments(argc, argv);

    std::string libexaudflibPath;
#ifdef CUSTOM_LIBEXAUDFLIB_PATH
        libexaudflibPath = std::string(CUSTOM_LIBEXAUDFLIB_PATH);
#else
        libexaudflibPath = std::string(::getenv("LIBEXAUDFLIB_PATH"));
#endif
        DBGMSG(std::cerr, "Load libexaudflib");
        DBGVAR(std::cerr, libexaudflibPath);
        void* handle = exa_load_libary(libexaudflibPath);
        if (!handle) {
            std::cerr << "Failed to load library: " << libexaudflibPath << std::endl;
            exit(EXIT_FAILURE);
        }

        MAIN_FUN exaudfclient_main = (MAIN_FUN)exa_load_symbol(handle, "exaudfclient_main");
        SET_SWIGVM_PARAMS set_SWIGVM_params = (SET_SWIGVM_PARAMS)exa_load_symbol(handle, "set_SWIGVM_params");

        setup_environment();
        std::function<SWIGVMContainers::SWIGVM*()> vmMaker = create_vm(m_languageArg, mb_useCtpgParser);

        SWIGVMContainers::SWIGVM_params = new SWIGVM_params_t(true);
        set_SWIGVM_params(SWIGVMContainers::SWIGVM_params);
        return exaudfclient_main(vmMaker, argc, argv);
}