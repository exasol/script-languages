#include <iostream>
#include <cstring>
#include <functional>
#include <string>

#include "exa_lib_loader.h"
#include "exa_parser_cfg.h"
#include "exa_set_env.h"
#include "exa_vm_factory.h"
#include "utils/debug_message.h"
#include "vm/swig_vm.h"
#include "exaudf_lib_output_path.h"

namespace SWIGVMContainers {
__thread SWIGVM_params_t * SWIGVM_params = nullptr;
}

typedef bool (*SET_SWIGVM_PARAMS)(SWIGVM_params_t*);
typedef int (*MAIN_FUN)(std::function<SWIGVMContainers::SWIGVM*()> vmMaker, int, char**);

void print_usage(const char *prg_name) {
    std::cerr   << "Usage: " << prg_name
                << " <socket> lang=python|lang=r|lang=java|lang=streaming|lang=benchmark <scriptOptionsParserVersion=1|2>"
                << std::endl;
}

bool validateArguments(int argc, char** argv) {
#ifdef UDF_PLUGIN_CLIENT
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <socket>" << std::endl;
        return false;
    }
    return true;
#else
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return false;
    }
    
    if (argc == 4 && 
        strcmp(argv[3], "scriptOptionsParserVersion=1") != 0 &&
        strcmp(argv[3], "scriptOptionsParserVersion=2") != 0) {
        print_usage(argv[0]);
        return false;
    }
    return true;
#endif
}

int main(int argc, char **argv) {
    if (!validateArguments(argc, argv)) {
        exit(EXIT_FAILURE);
    }
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
        fprintf(stderr, "Failed to load library: %s\n", libexaudflibPath.c_str());
        exit(EXIT_FAILURE);
    }

    MAIN_FUN exaudfclient_main = (MAIN_FUN)exa_load_symbol(handle, "exaudfclient_main");
    SET_SWIGVM_PARAMS set_SWIGVM_params = (SET_SWIGVM_PARAMS)exa_load_symbol(handle, "set_SWIGVM_params");

    bool is_use_ctpg_parser = is_use_ctpg_parser(argv[3]);

    setup_environment();
    std::function<SWIGVMContainers::SWIGVM*()>vmMaker = create_vm(argv[2]);

    SWIGVM_params = new SWIGVM_params_t(true);
    set_SWIGVM_params(SWIGVM_params);
    return exaudfclient_main(vmMaker, argc, argv);
}