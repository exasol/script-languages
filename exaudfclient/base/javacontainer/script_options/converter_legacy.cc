#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterLegacy::ConverterLegacy()
: Converter()
, m_jarPaths() {}

void ConverterLegacy::convertExternalJar(const std::string & value) {
    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        delim = value.find(":", start);
        if (delim != std::string::npos) {
            std::string jar = value.substr(start, delim - start);
            if (m_jarPaths.find(jar) == m_jarPaths.end())
                m_jarPaths.insert(jar);
        }
        else {
            std::string jar = value.substr(start);
            if (m_jarPaths.find(jar) == m_jarPaths.end())
                m_jarPaths.insert(jar);
            break;
        }
    }
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
