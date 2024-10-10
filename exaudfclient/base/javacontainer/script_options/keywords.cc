#include "base/javacontainer/script_options/keywords.h"
#include <string_view>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

Keywords::Keywords(bool withScriptOptionsPrefix)
: m_jarKeyword()
, m_scriptClassKeyword()
, m_importKeyword()
, m_jvmKeyword() {
    const std::string_view jar{"%jar"};
    const std::string_view scriptClass{"%scriptclass"};
    const std::string_view import{"%import"};
    const std::string_view jvm{"%jvmoption"};
    if (withScriptOptionsPrefix) {
        m_jarKeyword = jar;
        m_scriptClassKeyword = scriptClass;
        m_importKeyword = import;
        m_jvmKeyword = jvm;
    } else {
        m_jarKeyword.assign(jar.substr(1));
        m_scriptClassKeyword.assign(scriptClass.substr(1));
        m_importKeyword.assign(import.substr(1));
        m_jvmKeyword.assign(jvm.substr(1));
    }
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

