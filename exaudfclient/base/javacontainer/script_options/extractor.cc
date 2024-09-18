#include "base/javacontainer/script_options/extractor.h"
#include "base/javacontainer/script_options/parser_factory.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {



Extractor::Extractor(ParserFactory & parserFactory,
                     std::function<void(const std::string&)> throwException)
: m_throwException(throwException)
, m_parserFactory(parserFactory) {}


void Extractor::extract(std::string & scriptCode) {
    std::unique_ptr<ScriptOptionsParser> parser(m_parserFactory.makeParser());
    parser->prepareScriptCode(scriptCode);
    parser->parseForScriptClass( [&](const std::string& value){m_converter.convertScriptClassName(value);}, m_throwException);
    parser->extractImportScripts(m_throwException);
    parser->parseForJvmOptions( [&](const std::string& value){m_converter.convertScriptClassName(value);}, m_throwException);
    parser->parseForExternalJars( [&](const std::string& value){m_converter.convertScriptClassName(value);}, m_throwException);
    scriptCode = std::move(parser->getScriptCode());
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
