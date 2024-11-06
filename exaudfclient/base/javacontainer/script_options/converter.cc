#include "base/javacontainer/script_options/converter.h"

#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

Converter::Converter()
: m_jvmOptions()
, m_whitespace(" \t\f\v") {}

void Converter::convertScriptClassName(const std::string & value) {

    if (value.empty()) {
        return;
    }
    m_jvmOptions.push_back("-Dexasol.scriptclass=" + value);
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
