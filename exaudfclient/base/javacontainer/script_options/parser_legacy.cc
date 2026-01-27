#include "javacontainer/script_options/parser_legacy.h"
#include "javacontainer/script_options/distinct_script_set.h"
#include "script_options_parser/legacy/script_option_lines.h"
#include "exaudflib/swig/swig_meta_data.h"
#include "swig_factory/swig_factory.h"
#include "utils/exceptions.h"
#include "script_options_parser/exception.h"

#include <memory>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

ScriptOptionLinesParserLegacy::ScriptOptionLinesParserLegacy(std::unique_ptr<SwigFactory> swigFactory)
: m_whitespace(" \t\f\v")
, m_lineend(";")
, m_scriptCode()
, m_keywords(true)
, m_swigFactory(std::move(swigFactory)) {}

void ScriptOptionLinesParserLegacy::prepareScriptCode(const std::string & scriptCode) {
    m_scriptCode = scriptCode;
}

void ScriptOptionLinesParserLegacy::extractImportScripts() {
    std::unique_ptr<SWIGMetadataIf> metaData;
    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports its self
    DistinctScriptSet importedScripts;
    importedScripts.addScript(m_scriptCode.c_str());
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
    because the content of "other_script_A" is already stored in the DistinctScriptSet,
    and the parser only removes the script options keyword, but does not insert again the code of other_script_A.
    The fourth iteration of the while loop would detect no further import option keywords and would break the loop.
    */
    while (true) {
        std::string newScript;
        size_t scriptPos;
        try {
            parseForSingleOption(m_keywords.importKeyword(),
                                 [&](const std::string& value, size_t pos){scriptPos = pos; newScript = value;});
        } catch (const ExecutionGraph::OptionParserException & ex) {
            Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1614");
        }
        if (!newScript.empty()) {
            if (!metaData) {
                metaData.reset(m_swigFactory->makeSwigMetadata());
                if (!metaData)
                    throw std::runtime_error("F-UDF-CL-SL-JAVA-1615: Failure while importing scripts");
            }
            const char *importScriptCode = metaData->moduleContent(newScript.c_str());
            const char *exception = metaData->checkException();
            if (exception)
                throw std::runtime_error("F-UDF-CL-SL-JAVA-1616: " + std::string(exception));
            if (importedScripts.addScript(importScriptCode)) {
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

void ScriptOptionLinesParserLegacy::parseForScriptClass(std::function<void(const std::string &option)> callback) {
    try {
    parseForSingleOption(m_keywords.scriptClassKeyword(),
                            [&](const std::string& value, size_t pos){callback(value);});
    } catch (const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1610");
    }
}

void ScriptOptionLinesParserLegacy::parseForJvmOptions(std::function<void(const std::string &option)> callback) {
    try {
       parseForMultipleOptions(m_keywords.jvmKeyword(),
                                [&](const std::string& value, size_t pos){callback(value);});
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1612");
    }
}

void ScriptOptionLinesParserLegacy::parseForExternalJars(std::function<void(const std::string &option)> callback) {
    try {
       parseForMultipleOptions(m_keywords.jarKeyword(),
                                [&](const std::string& value, size_t pos){callback(value);});
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1613");
    }
}

std::string && ScriptOptionLinesParserLegacy::getScriptCode() {
    return std::move(m_scriptCode);
}

void ScriptOptionLinesParserLegacy::parseForSingleOption(const std::string & keyword,
                            std::function<void(const std::string &option, size_t pos)> callback) {
    size_t pos;
    try {
        const std::string option =
          ExecutionGraph::extractOptionLine(m_scriptCode, keyword, m_whitespace, m_lineend, pos);
        if (option != "") {
            callback(option, pos);
        }
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1606");
    }
}

void ScriptOptionLinesParserLegacy::parseForMultipleOptions(const std::string & keyword,
                            std::function<void(const std::string &option, size_t pos)> callback) {
    size_t pos;
    while (true) {
        try {
            const std::string option =
              ExecutionGraph::extractOptionLine(m_scriptCode, keyword, m_whitespace, m_lineend, pos);
            if (option == "")
                break;
            callback(option, pos);
        } catch(const ExecutionGraph::OptionParserException& ex) {
            Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1607");
        }
    }
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
