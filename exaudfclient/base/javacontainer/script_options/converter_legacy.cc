#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>
#include <sstream>

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

void ConverterLegacy::iterateJarPaths(Converter::tJarIteratorCallback callback) const {
    for (const auto & jar: m_jarPaths) {
        callback(jar);
    }
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers