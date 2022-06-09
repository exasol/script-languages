#include "exaudflib/exaudflib_main.h"

/*
 The purpose of this function is only to force the linker to include all necessary dependencies into libexaudflib_complete.so
*/
void exaudfclient_main_dummy() {
  std::function<SWIGVMContainers::SWIGVM*()>vmMaker=[](){return nullptr;};
  exaudfclient_main(vmMaker, 0, nullptr);
}
