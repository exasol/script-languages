#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterV2::ConverterV2()
: Converter()
, m_jarPaths() {}

void ConverterV2::convertExternalJar(const std::string & value) {
    std::istringstream stream(value);
    std::string jar;

    while (std::getline(stream, jar, ':')) {
        m_jarPaths.push_back(jar);
    }
}

void ConverterV2::iterateJarPaths(Converter::tJarIteratorCallback callback) const {
    std::for_each(m_jarPaths.begin(), m_jarPaths.end(), callback);
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
