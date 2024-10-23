#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

Converter::Converter()
: m_whitespace(" \t\f\v")
, m_jvmOptions() {}

void Converter::convertScriptClassName(const std::string & value) {
    std::string trimmedValue(value);
    StringOps::trim(trimmedValue);
    if (value != "") {
        m_jvmOptions.push_back("-Dexasol.scriptclass=" + trimmedValue);
    }
}

void Converter::convertJvmOption(const std::string & value) {
    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        start = value.find_first_not_of(m_whitespace, start);
        if (start == std::string::npos)
            break;
        delim = value.find_first_of(m_whitespace, start);
        if (delim != std::string::npos) {
            m_jvmOptions.push_back(value.substr(start, delim - start));
        }
        else {
            m_jvmOptions.push_back(value.substr(start));
            break;
        }
    }
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
