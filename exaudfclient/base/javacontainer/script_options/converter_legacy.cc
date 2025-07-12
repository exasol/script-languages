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

void ConverterLegacy::convertJvmOption(const std::string& value) {
    std::string::size_type start = 0, delim = 0;
    while ((start = value.find_first_not_of(m_whitespace, delim)) != std::string::npos) {
        delim = value.find_first_of(m_whitespace, start);
        const std::string::size_type len = (std::string::npos == delim) ?  std::string::npos : delim - start;
        m_jvmOptions.push_back(value.substr(start, len));
    }
}


void ConverterLegacy::iterateJarPaths(Converter::tJarIteratorCallback callback) const {
    std::for_each(m_jarPaths.begin(), m_jarPaths.end(), callback);
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
