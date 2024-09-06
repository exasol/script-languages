#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>


#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/parser.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Extractor() {

    public:
        Extractor(const std::string scriptCode,
                    std::function<void(const std::string&)> throwException);

        const std::string& getModifiedScriptCode() const {
            return m_modifiedCode;
        }

        const std::set<std::string> & getJarPaths() const {
            return m_converter.getJarPaths();
        }

        const std::vector<std::string>& getJvmOptions() const {
            return m_converter.getJvmOptions();
        }

        const std::string& getScriptClassName() const {
            return m_converter.getScriptClassName();
        }

        void extract();
    private:
        ScriptOptionsParser* makeParser(std::string & scriptCode);


        void extractImportScripts(ScriptOptionsParser* parser);

        void extractImportScript(SWIGMetadata** metaData, std::string & scriptCode, ScriptOptionsParser *parser,
                                    std::set<std::vector<unsigned char> > & importedScriptChecksums);

    private:
        std::string m_modifiedCode;

        std::function<void(const std::string&)> m_throwException;

        std::string m_jarKeyword;
        std::string m_scriptClassKeyword;
        std::string m_importKeyword;
        std::string m_jvmOptionKeyword;

        Converter m_converter;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H