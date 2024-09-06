#include "base/javacontainer/script_options/converter.h"

namespace SWIGVMContainers {

namespace JavaScriptOptions {



ScriptOptionsConverter::ScriptOptionsConverter(std::function<void(const std::string&)> throwException,
                                               std::vector<std::string>& jvmOptions):
m_throwException(throwException),
m_jvmOptions(),
m_jarPaths(),
m_scriptClassName()
{}


void ScriptOptionsConverter::convertExternalJar(const std::string & value) {
    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        delim = value.find(":", start);
        if (delim != std::string::npos) {
            std::string jar = jarPath.substr(start, delim - start);
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

void ScriptOptionsConverter::convertScriptClassName(const std::string & value) {
    if (value != "") {
        m_jvmOptions.push_back("-Dexasol.scriptclass=" + value);
    }
}

void ScriptOptionsConverter::convertJvmOption(const std::string & value) {
    for (size_t start = 0, delim = 0; ; start = delim + 1) {
        start = value.find_first_not_of(whitespace, start);
        if (start == std::string::npos)
            break;
        delim = jvmOption.find_first_of(whitespace, start);
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
