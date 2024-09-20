#ifndef SCRIPTOPTIONLINEKEYWORDS_H
#define SCRIPTOPTIONLINEKEYWORDS_H 1

#include <string>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Keywords {
    public:
        Keywords()
            : m_jarKeyword("%jar")
            , m_scriptClassKeyword("%scriptclass")
            , m_importKeyword("%import")
            , m_jvmOptionKeyword("%jvmoption") {}
        const std::string & jarKeyword() { return m_jarKeyword; }
        const std::string & scriptClassKeyword() { return m_scriptClassKeyword; }
        const std::string & importKeyword() { return m_importKeyword; }
        const std::string & jvmOptionKeyword() { return m_jvmOptionKeyword; }
    private:
        std::string m_jarKeyword;
        std::string m_scriptClassKeyword;
        std::string m_importKeyword;
        std::string m_jvmOptionKeyword;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif // SCRIPTOPTIONLINEKEYWORDS_H