#pragma once
#include "swig_vm.h"

#ifdef ENABLE_JAVA_VM
#include "javacontainer/javacontainer_builder.h"
#endif //ENABLE_JAVA_VM

#ifdef ENABLE_PYTHON_VM
#include "python/pythoncontainer.h"
#endif //ENABLE_PYTHON_VM

#ifdef UDF_PLUGIN_CLIENT
#include "protegrityclient.h"
#endif //UDF_PLUGIN_CLIENT

#ifdef ENABLE_STREAMING_VM
#include "streaming_container/streamingcontainer.h"
#endif

#ifdef ENABLE_BENCHMARK_VM
#include "benchmark_container/benchmark_container.h"
#endif

std::function<SWIGVMContainers::SWIGVM*()> create_vm(const std::string& argv_lang, bool use_ctpg_options_parser);