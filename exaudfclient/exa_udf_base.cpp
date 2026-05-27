#include <iostream>
#include <string>

#include "exaudf_lib_output_path.h"
#include "vm/swig_vm.h"
#include "exa_udf_base.h"
#include "exa_lib_loader.h"


void ExaUdfClientBase::parse_arguments(int argc, char** argv) {
    //  Now assumption is that all cmd line aguments are already validated.
    m_socket = argv[1];
    std::string strLangParam = argv[2];

    if(strLangParam.find("python") != std::string::npos)
        m_lang = ExaUdfLanguage::Python3;
    else if(strLangParam.find("java") != std::string::npos)
        m_lang = ExaUdfLanguage::Java;
    else if(strLangParam.find("streaming") != std::string::npos)
        m_lang = ExaUdfLanguage::Streaming;
    else if(strLangParam.find("benchmark") != std::string::npos)
        m_lang = ExaUdfLanguage::Benchmark;
   
    mb_useCtpgParser = false;
    if(argc == 4) {
        mb_useCtpgParser = is_use_ctpg_parser(argv[3]);
    }
}

//  Parser option 2 means use ctpg parser.
//  argv_parser_option is command line argument as it is
bool ExaUdfClientBase::is_use_ctpg_parser(const std::string& argv_parser_option) {
    bool use_ctpg_option_parser = false;

    //  env var has higher priority than argv value.
    const char* env_val = ::getenv("SCRIPT_OPTIONS_PARSER_VERSION");
    if(env_val) {
        use_ctpg_option_parser = (strcmp(env_val, "2") == 0);
    }
    else if(argv_parser_option.compare("scriptOptionsParserVersion=2") == 0) {
        use_ctpg_option_parser = true;
    }
    return use_ctpg_option_parser;    
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
        std::function<SWIGVMContainers::SWIGVM*()> vmMaker;
        switch(m_lang) {
            case ExaUdfLanguage::Python3:
                ExaUdfClientPython pythonClient;
                vmMaker = pythonClient.create_vm();
                break;
            case ExaUdfLanguage::Java:
                ExaUdfClientJava javaClient;
                vmMaker = javaClient.create_vm();
                break;
            case ExaUdfLanguage::Streaming:
                ExaUdfClientStreaming streamingClient;
                vmMaker = streamingClient.create_vm();
                break;
            case ExaUdfLanguage::Benchmark:
                ExaUdfClientBenchmark benchmarkClient;
                vmMaker = benchmarkClient.create_vm();
                break;
        }

        SWIGVMContainers::SWIGVM_params = new SWIGVMContainers::SWIGVM_params_t(true);
        set_SWIGVM_params(SWIGVMContainers::SWIGVM_params);
        return exaudfclient_main(vmMaker, argc, argv);
}