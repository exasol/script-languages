#include "exaudflib/swig/swig_common.h"

//Dummy implementation to calm down the linker
void* load_dynamic(const char* name) {
    return nullptr;
}

//Globalinstance of the SWIGVM_params which we used to communicate with JavaVMImpl;
namespace SWIGVMContainers {
SWIGVM_params_t gSWIGVM_params_t;
__thread SWIGVM_params_t * SWIGVM_params = &gSWIGVM_params_t;
}