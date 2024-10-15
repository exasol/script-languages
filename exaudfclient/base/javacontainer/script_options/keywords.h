#ifndef SCRIPTOPTIONLINEKEYWORDS_H
#define SCRIPTOPTIONLINEKEYWORDS_H 1

#include <string>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Keywords {
    public:
        Keywords(bool withScriptOptionsPrefix);
        const std::string & scriptClassKeyword() { return m_scriptClassKeyword; }
        const std::string & importKeyword() { return m_importKeyword; }
        const std::string & jvmKeyword() { return m_jvmKeyword; }
        const std::string & jarKeyword() { return m_jarKeyword; }
    private:
        std::string m_jarKeyword;
        std::string m_scriptClassKeyword;
        std::string m_importKeyword;
        std::string m_jvmKeyword;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif // SCRIPTOPTIONLINEKEYWORDS_H