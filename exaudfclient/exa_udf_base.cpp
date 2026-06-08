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
        std::function<SWIGVMContainers::SWIGVM*()> vmMaker = create_vm();

        SWIGVMContainers::SWIGVM_params = new SWIGVM_params_t(true);
        set_SWIGVM_params(SWIGVMContainers::SWIGVM_params);
        return exaudfclient_main(vmMaker, argc, argv);
}