#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/script_options_parser/script_option_lines.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

ScriptOptionLinesParserLegacy::ScriptOptionLinesParserLegacy(std::string & scriptCode)
: m_whitespace(" \t\f\v")
, m_lineend(";")
, m_scriptCode(scriptCode) {}

void ScriptOptionLinesParserLegacy::parseForSingleOption(const std::string keyword,
                            std::function<void(const std::string &option, size_t pos)> callback,
                            std::function<void(const std::string&)> throwException) {
    size_t pos;
    const std::string option =
      ExecutionGraph::extractOptionLine(
          m_scriptCode,
          keyword,
          m_whitespace,
          m_lineend,
          pos,
          [&](const char* msg){throwException(std::string("F-UDF-CL-SL-JAVA-1606: ") + msg);}
          );
    if (option != "") {
        callback(option, pos);
    }
}


void ScriptOptionLinesParserLegacy::parseForMultipleOptions(const std::string keyword,
                            std::function<void(const std::string &option, size_t pos)> callback,
                            std::function<void(const std::string&)> throwException) {
    size_t pos;
    while (true) {
        const std::string option =
          ExecutionGraph::extractOptionLine(
              m_scriptCode,
              keyword,
              m_whitespace,
              m_lineend,
              pos,
              [&](const char* msg){throwException(std::string("F-UDF-CL-SL-JAVA-1607: ") + msg);}
              );
        if (option == "")
            break;
        callback(option, pos);
    }
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
