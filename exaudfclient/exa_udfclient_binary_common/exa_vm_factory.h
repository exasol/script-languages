#pragma once
#include "exaudflib/vm/swig_vm.h"

std::function<SWIGVMContainers::SWIGVM*()> create_vm(const std::string& argv_lang, bool use_ctpg_options_parser);