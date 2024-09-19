#include "base/javacontainer/script_options/converter.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

Converter::Converter()
: m_jvmOptions()
, m_jarPaths()
, m_whitespace(" \t\f\v") {}

void Converter::convertExternalJar(const std::string & value) {
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

void Converter::convertScriptClassName(const std::string & value) {
    if (value != "") {
        m_jvmOptions.push_back("-Dexasol.scriptclass=" + value);
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
