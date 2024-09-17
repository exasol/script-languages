#include "base/javacontainer/script_options/parser_factory.h"
#include "base/javacontainer/script_options/parser_legacy.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

std::unique_ptr<ScriptOptionsParser> ParserFactoryLegacy::makeParser() {
    return std::make_unique<ScriptOptionLinesParserLegacy>();
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
