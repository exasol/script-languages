#include "base/javacontainer/script_options/parser_factory.h"
#include "base/javacontainer/script_options/parser_legacy.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

std::unique_ptr<ScriptOptionsParser> ParserFactoryLegacy::makeParser(std::string & scriptCode) {
    return std::make_unique<ScriptOptionLinesParserLegacy>(scriptCode);
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
