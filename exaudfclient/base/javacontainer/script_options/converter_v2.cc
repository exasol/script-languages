#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterV2::ConverterV2()
: Converter()
, m_jarPaths() {}

void ConverterV2::convertExternalJar(const std::string & value) {
    std::string formattedValue(value);
    StringOps::trim(formattedValue);
    if (formattedValue.size() > 1 && formattedValue.front() == '\"' && formattedValue.back() == '\"') {
        formattedValue = formattedValue.substr(1, formattedValue.size()-2);
    }

    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        delim = formattedValue.find(":", start);
        if (delim != std::string::npos) {
            std::string jar = formattedValue.substr(start, delim - start);
            if (m_jarPaths.find(jar) == m_jarPaths.end())
                m_jarPaths.insert(jar);
        }
        else {
            std::string jar = formattedValue.substr(start);
            if (m_jarPaths.find(jar) == m_jarPaths.end())
                m_jarPaths.insert(jar);
            break;
        }
    }
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
