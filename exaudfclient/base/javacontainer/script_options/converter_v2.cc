#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>
#include <istream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterV2::ConverterV2()
: Converter()
, m_jarPaths() {}

void ConverterV2::convertExternalJar(const std::string & value) {
    std::istringstream stream(value);
    std::string jar;

    while (std::getline(stream, jar, ':')) {
        m_jarPaths.insert(jar); // If m_jarPaths is a Set or you make it a Set, uniqueness implicitly is taken care of
    }
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
