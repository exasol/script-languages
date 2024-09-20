#ifndef SCRIPTOPTIONLINEPARSERLEGACY_H
#define SCRIPTOPTIONLINEPARSERLEGACY_H 1


#include "base/javacontainer/script_options/parser.h"
#include "base/javacontainer/script_options/keywords.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ScriptOptionLinesParserLegacy : public ScriptOptionsParser {

    public:
        ScriptOptionLinesParserLegacy();

        void prepareScriptCode(const std::string & scriptCode) override;

        void parseForScriptClass(std::function<void(const std::string &option)> callback,
                                 std::function<void(const std::string&)> throwException) override;

        void parseForJvmOptions(std::function<void(const std::string &option)> callback,
                                std::function<void(const std::string&)> throwException) override;

        void parseForExternalJars(std::function<void(const std::string &option)> callback,
                                  std::function<void(const std::string&)> throwException) override;

        void extractImportScripts(std::function<void(const std::string&)> throwException) override;

        std::string && getScriptCode() override;

    private:
        void parseForSingleOption(const std::string key,
                                        std::function<void(const std::string &option, size_t pos)> callback,
                                        std::function<void(const std::string&)> throwException);
        void parseForMultipleOptions(const std::string key,
                                                std::function<void(const std::string &option, size_t pos)> callback,
                                                std::function<void(const std::string&)> throwException);

    private:
        const std::string m_whitespace;
        const std::string m_lineend;
        std::string m_scriptCode;
        Keywords m_keywords;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSERLEGACY_H