#ifndef SCRIPTOPTIONLINEPARSERLEGACY_H
#define SCRIPTOPTIONLINEPARSERLEGACY_H 1


#include "base/javacontainer/script_options/parser.h"
#include "base/javacontainer/script_options/keywords.h"

#include <memory>

namespace SWIGVMContainers {

struct SwigFactory;

namespace JavaScriptOptions {

class ScriptOptionLinesParserLegacy : public ScriptOptionsParser {

    public:
        ScriptOptionLinesParserLegacy(std::unique_ptr<SwigFactory> swigFactory);

        virtual ~ScriptOptionLinesParserLegacy() {};

        void prepareScriptCode(const std::string & scriptCode) override;

        void parseForScriptClass(std::function<void(const std::string &option)> callback) override;

        void parseForJvmOptions(std::function<void(const std::string &option)> callback) override;

        void parseForExternalJars(std::function<void(const std::string &option)> callback) override;

        void extractImportScripts() override;

        std::string && getScriptCode() override;

    private:
        void parseForSingleOption(const std::string& key,
                                        std::function<void(const std::string &option, size_t pos)> callback);
        void parseForMultipleOptions(const std::string& key,
                                                std::function<void(const std::string &option, size_t pos)> callback);

    private:
        const std::string m_whitespace;
        const std::string m_lineend;
        std::string m_scriptCode;
        Keywords m_keywords;
        std::unique_ptr<SwigFactory> m_swigFactory;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSERLEGACY_H