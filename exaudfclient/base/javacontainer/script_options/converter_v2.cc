#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterV2::ConverterV2()
: Converter()
, m_jarPaths() {}

void ConverterV2::convertExternalJar(const std::string & value) {
    std::string unquotedValue(value);
    StringOps::removeQuotesSafely(unquotedValue, m_whitespace);
    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        delim = unquotedValue.find(":", start);
        if (delim != std::string::npos) {
            std::string jar = unquotedValue.substr(start, delim - start);
            StringOps::removeQuotesSafely(jar, m_whitespace);
            m_jarPaths.push_back(jar);
        }
        else {
            std::string jar = unquotedValue.substr(start);
            StringOps::removeQuotesSafely(jar, m_whitespace);
            m_jarPaths.push_back(jar);
            break;
        }
    }
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
