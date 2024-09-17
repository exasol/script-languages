#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>


#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/parser.h"


namespace SWIGVMContainers {

class SWIGMetadata;

namespace JavaScriptOptions {

class ParserFactory;

class Extractor {

    public:
        Extractor(const std::string & scriptCode,
                    ParserFactory & parserFactory,
                    std::function<void(const std::string&)> throwException);

        std::string&& moveModifiedScriptCode() {
            return std::move(m_modifiedCode);
        }

        const std::set<std::string> & getJarPaths() const {
            return m_converter.getJarPaths();
        }

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_converter.moveJvmOptions());
        }

        void extract();
    private:
        std::unique_ptr<ScriptOptionsParser> makeParser();


        void extractImportScripts(ScriptOptionsParser* parser);

        void extractImportScript(std::unique_ptr<SWIGMetadata>& metaData, std::string & scriptCode,
                                        const std::string &importScriptId, size_t importScriptPos,
                                        std::set<std::vector<unsigned char> > & importedScriptChecksums);

    private:
        std::string m_modifiedCode;

        std::function<void(const std::string&)> m_throwException;

        std::string m_jarKeyword;
        std::string m_scriptClassKeyword;
        std::string m_importKeyword;
        std::string m_jvmOptionKeyword;

        Converter m_converter;
        ParserFactory & m_parserFactory;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H