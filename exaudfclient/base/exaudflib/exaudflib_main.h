#ifndef EXAUDFLIB_MAIN_H
#define EXAUDFLIB_MAIN_H

#include "exaudflib/vm/swig_vm.h"
#include <functional>

extern "C" {

int exaudfclient_main(std::function<SWIGVMContainers::SWIGVM*()>vmMaker,int argc,char**argv);

}

#endif //EXAUDFLIB_MAIN_H