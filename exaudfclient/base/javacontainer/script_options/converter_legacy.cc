#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterLegacy::ConverterLegacy()
: Converter()
, m_jarPaths() {}

void ConverterLegacy::convertExternalJar(const std::string& value) {
    std::istringstream stream(value);
    std::string jar;

    while (std::getline(stream, jar, ':')) {
        m_jarPaths.insert(jar);
    }
}

void ConverterLegacy::convertJvmOption(const std::string & value) {
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


void ConverterLegacy::iterateJarPaths(Converter::tJarIteratorCallback callback) const {
    std::for_each(m_jarPaths.begin(), m_jarPaths.end(), callback);
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
