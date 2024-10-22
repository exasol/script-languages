#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

inline uint32_t countBackslashesBackwards(const  std::string & s, size_t start) {
    uint32_t retVal = 0;
    if (start < s.size() && start >= 0) {
        while (start >= 0 && s[start--] == '\\') {
            retVal++;
        }
    }
    return retVal;
}

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

void Converter::convertExternalJarWithEscapeSequences(const std::string & value) {
    std::string formattedValue(value);
    StringOps::trim(formattedValue);
    if (formattedValue.size() > 1 && formattedValue.front() == '\"' && formattedValue.back() == '\"') {
        formattedValue = formattedValue.substr(1, formattedValue.size()-2);
    }

    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        size_t search_start = start;
        do {
            delim = formattedValue.find(":", search_start);
            search_start = delim + 1;
        } while (delim != std::string::npos && delim != 0 && countBackslashesBackwards(formattedValue, delim-1) % 2 == 1);

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
