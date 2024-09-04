#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/script_options_parser/scriptoptionlines.h"


namespace SWIGVMContainers {

namespace JavaScriptOptions {

ScriptOptionLinesParserLegacy::ScriptOptionLinesParserLegacy()
: m_whitespace(" \t\f\v")
, m_lineend(";")
, m_jarKeyword("%jar")
, m_scriptClassKeyword("%scriptclass")
, m_importKeyword("%import")
, m_jvmOptionKeyword("%jvmoption") {}

void ScriptOptionLinesParserLegacy::findExternalJarPaths(std::string & src_scriptCode,
                                                             std::vector<std::string>& jarPaths,
                                                             std::function<void(const std::string&)> throwException) {
    callParserForManyValues(src_scriptCode, m_jarKeyword, jarPaths, throwException);
}


void ScriptOptionLinesParserLegacy::getScriptClassName(std::string & src_scriptCode, std::string &scriptClassName,
                                                       std::function<void(const std::string&)> throwException) {
    size_t pos;
    scriptClassName = callParserForSingleValue(src_scriptCode, pos, m_scriptClassKeyword, throwException);
}


void ScriptOptionLinesParserLegacy::getNextImportScript(std::string & src_scriptCode,
                                                        std::pair<std::string, size_t> & result,
                                                        std::function<void(const std::string&)> throwException) {
    size_t pos;
    const std::string scriptName = callParserForSingleValue(src_scriptCode, pos, m_importKeyword, throwException);
    result.first = scriptName;
    result.second = pos;
}


void ScriptOptionLinesParserLegacy::getExternalJvmOptions(std::string & src_scriptCode,
                                                              std::vector<std::string>& jvmOptions,
                                                              std::function<void(const std::string&)> throwException) {
    callParserForManyValues(src_scriptCode, m_jvmOptionKeyword, jvmOptions, throwException);
}


std::string ScriptOptionLinesParserLegacy::callParserForSingleValue(std::string & src_scriptCode, size_t & pos,
                                                                    const std::string & keyword,
                                                                    std::function<void(const std::string&)> throwException) {
    return ExecutionGraph::extractOptionLine(
          src_scriptCode,
          keyword,
          m_whitespace,
          m_lineend,
          pos,
          [&](const char* msg){throwException(std::string("F-UDF-CL-SL-JAVA-1606: ") + msg);}
          );
}

void ScriptOptionLinesParserLegacy::callParserForManyValues(std::string & src_scriptCode,
                                                            const std::string & keyword,
                                                            std::vector<std::string>& result,
                                                            std::function<void(const std::string&)> throwException) {
    size_t pos;
    while (true) {
        const std::string options =
          ExecutionGraph::extractOptionLine(
              src_scriptCode,
              keyword,
              m_whitespace,
              m_lineend,
              pos,
              [&](const char* msg){throwException(std::string("F-UDF-CL-SL-JAVA-1607: ") + msg);}
              );
        if (options == "")
            break;
        result.push_back(options);
    }
}



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
