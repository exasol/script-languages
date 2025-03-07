#ifndef SCRIPTOPTIONLINEPARSERCTPGY_H
#define SCRIPTOPTIONLINEPARSERCTPGY_H 1

#include "base/javacontainer/script_options/parser.h"
#include "base/javacontainer/script_options/keywords.h"
#include "base/script_options_parser/ctpg/script_option_lines_ctpg.h"
#include <memory>

namespace SWIGVMContainers {

struct SwigFactory;

namespace JavaScriptOptions {

class ScriptOptionLinesParserCTPG : public ScriptOptionsParser {

    public:
        ScriptOptionLinesParserCTPG(std::unique_ptr<SwigFactory> swigFactory);

        virtual ~ScriptOptionLinesParserCTPG() {};

        void prepareScriptCode(const std::string & scriptCode) override;

        void parseForScriptClass(std::function<void(const std::string &option)> callback) override;

        void parseForJvmOptions(std::function<void(const std::string &option)> callback) override;

        void parseForExternalJars(std::function<void(const std::string &option)> callback) override;

        void extractImportScripts() override;

        std::string && getScriptCode() override;

    private:
        void parse();

        void parseForSingleOption(const std::string key, std::function<void(const std::string &option)> callback);
        void parseForMultipleOption(const std::string key, std::function<void(const std::string &option)> callback);

        void importScripts(SwigFactory & swigFactory);

    private:
        std::string m_scriptCode;
        Keywords m_keywords;
        ExecutionGraph::OptionsLineParser::CTPG::options_map_t m_foundOptions;
        bool m_needParsing;
        std::unique_ptr<SwigFactory> m_swigFactory;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCTPGY_H
