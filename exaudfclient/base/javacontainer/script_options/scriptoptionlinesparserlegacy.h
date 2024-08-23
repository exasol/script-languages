#ifndef SCRIPTOPTIONLINEPARSERLEGACY_H
#define SCRIPTOPTIONLINEPARSERLEGACY_H 1


#include "base/javacontainer/script_options/scriptoptionlinesparser.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ScriptOptionLinesParserLegacy : public ScriptOptionsParser {

    public:
        ScriptOptionLinesParserLegacy();

        void findExternalJarPaths(std::string & src_scriptCode,
                                  std::vector<std::string>& jarPaths,
                                  std::function<void(const std::string&)> throwException);

        void getScriptClassName(std::string & src_scriptCode, std::string &scriptClassName,
                                std::function<void(const std::string&)> throwException);

        void getNextImportScript(std::string & src_scriptCode,
                                 std::pair<std::string, size_t> & result,
                                 std::function<void(const std::string&)> throwException);


        void getExternalJvmOptions(std::string & src_scriptCode,
                                   std::vector<std::string>& jvmOptions,
                                   std::function<void(const std::string&)> throwException);

    private:
        std::string  callParserForSingleValue(std::string & src_scriptCode, size_t &pos, const std::string & keyword,
                                              std::function<void(const std::string&)> throwException);
        void  callParserForManyValues(std::string & src_scriptCode, const std::string & keyword,
                                      std::vector<std::string>& result,
                                      std::function<void(const std::string&)> throwException);
    private:
        const std::string m_whitespace;
        const std::string m_lineend;
        const std::string m_jarKeyword;
        const std::string m_scriptClassKeyword;
        const std::string m_importKeyword;
        const std::string m_jvmOptionKeyword;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSERLEGACY_H