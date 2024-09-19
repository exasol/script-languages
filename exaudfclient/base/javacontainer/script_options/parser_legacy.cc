#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/javacontainer/script_options/checksum.h"
#include "base/script_options_parser/script_option_lines.h"
#include "base/exaudflib/swig/swig_meta_data.h"

#include <memory>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ScriptOptionLinesParserLegacy::ScriptOptionLinesParserLegacy()
: m_whitespace(" \t\f\v")
, m_lineend(";")
, m_scriptCode()
, m_keywords() {}

void ScriptOptionLinesParserLegacy::prepareScriptCode(const std::string & scriptCode) {
    m_scriptCode = scriptCode;
}

void ScriptOptionLinesParserLegacy::extractImportScripts(std::function<void(const std::string&)> throwException) {
    std::unique_ptr<SWIGMetadata> metaData;
    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports its self
    Checksum importedScriptChecksums;
    importedScriptChecksums.addScript(m_scriptCode.c_str());
    /*
    The following while loop iteratively replaces import scripts in the script code. Each replacement is done in two steps:
    1. Remove the "%import ..." option in the script code
    2. Insert the new script code at the same location
    It can happen that the imported scripts have again an "%import ..." option.
    Those cases will be handled in the next iteration of the while loop, because the parser searches the options in
    each iteration from the beginning of the (then modified) script code.
    For example, lets assume the following Jave UDF, stored in member variable 'm_scriptCode':
        %import other_script_A;
        class MyJavaUdf {
            static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(\"Success!\");\n"
            }
        };
    and 'other_script_A' is defined as (which will be retrieved over SWIGMetadata.moduleContent()):
        %import other_script_B;
        class OtherClassA {
            static void doSomething() {
            }
        };
    and other_script_B as:
        %import other_script_A;
        class OtherClassB {
            static void doSomething() {
            }
        };
    The first iteration of the while loop would modify the member variable m_scriptCode to:
        %import other_script_B;
        class OtherClassA {
            static void doSomething() {
            }
        };
        class MyJavaUdf {
            static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(\"Success!\");\n"
            }
        };
    The second iteration of the while loop would modify the member variable m_scriptCode to:
        The first iteration of the while loop would modify the member variable m_scriptCode to:
        %import other_script_A;
        class OtherClassB {
            static void doSomething() {
            }
        };
        class OtherClassA {
            static void doSomething() {
            }
        };
        class MyJavaUdf {
            static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(\"Success!\");\n"
            }
        };
    The third iteration of the while loop would modify the member variable m_scriptCode to:
        class OtherClassB {
            static void doSomething() {
            }
        };
        class OtherClassA {
            static void doSomething() {
            }
        };
        class MyJavaUdf {
            static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(\"Success!\");\n"
            }
        };
    because the content of "other_script_A" is already stored in the checksum,
    and the parser only removes the script options keyword, but does not insert again the code of other_script_A.
    The fourth iteration of the while loop would detect no further import option keywords and would break the loop.
    */
    while (true) {
        std::string newScript;
        size_t scriptPos;
        parseForSingleOption(m_keywords.importKeyword(),
                             [&](const std::string& value, size_t pos){scriptPos = pos; newScript = value;},
                             [&](const std::string& msg){throwException("F-UDF-CL-SL-JAVA-1614" + msg);});
        if (!newScript.empty()) {
            if (!metaData) {
                metaData = std::make_unique<SWIGMetadata>();
                if (!metaData)
                    throwException("F-UDF-CL-SL-JAVA-1615: Failure while importing scripts");
            }
            const char *importScriptCode = metaData->moduleContent(newScript.c_str());
            const char *exception = metaData->checkException();
            if (exception)
                throwException("F-UDF-CL-SL-JAVA-1616: " + std::string(exception));
            if (importedScriptChecksums.addScript(importScriptCode)) {
                // Script has not been imported yet
                // If this imported script contains %import statements
                // they will be resolved in the next iteration of the while loop.
                m_scriptCode.insert(scriptPos, importScriptCode);
            }
        } else {
            break;
        }
    }
}

void ScriptOptionLinesParserLegacy::parseForScriptClass(std::function<void(const std::string &option)> callback,
                                 std::function<void(const std::string&)> throwException) {
    parseForSingleOption(m_keywords.scriptClassKeyword(),
                            [&](const std::string& value, size_t pos){callback(value);},
                            [&](const std::string& msg){throwException("F-UDF-CL-SL-JAVA-1610" + msg);});
}

void ScriptOptionLinesParserLegacy::parseForJvmOptions(std::function<void(const std::string &option)> callback,
                                 std::function<void(const std::string&)> throwException) {
   parseForMultipleOptions(m_keywords.jvmOptionKeyword(),
                            [&](const std::string& value, size_t pos){callback(value);},
                            [&](const std::string& msg){throwException("F-UDF-CL-SL-JAVA-1612" + msg);});
}

void ScriptOptionLinesParserLegacy::parseForExternalJars(std::function<void(const std::string &option)> callback,
                                 std::function<void(const std::string&)> throwException) {
   parseForMultipleOptions(m_keywords.jarKeyword(),
                            [&](const std::string& value, size_t pos){callback(value);},
                            [&](const std::string& msg){throwException("F-UDF-CL-SL-JAVA-1613" + msg);});
}

std::string && ScriptOptionLinesParserLegacy::getScriptCode() {
    return std::move(m_scriptCode);
}

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
