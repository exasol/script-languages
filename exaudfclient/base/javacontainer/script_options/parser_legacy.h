#ifndef SCRIPTOPTIONLINEPARSERLEGACY_H
#define SCRIPTOPTIONLINEPARSERLEGACY_H 1


#include "base/javacontainer/script_options/parser.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ScriptOptionLinesParserLegacy : public ScriptOptionsParser {

    public:
        ScriptOptionLinesParserLegacy();

        virtual void parseForSingleOption(std::string & scriptCode, const std::string key,
                                        std::function<void(const std::string &option, size_t pos)> callback,
                                        std::function<void(const std::string&)> throwException);
        virtual void parseForMultipleOptions(std::string & scriptCode, const std::string key,
                                                std::function<void(const std::string &option, size_t pos)> callback,
                                                std::function<void(const std::string&)> throwException);

    private:
        const std::string m_whitespace;
        const std::string m_lineend;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSERLEGACY_H