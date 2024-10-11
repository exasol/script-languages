#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_ctpg_script_importer.h"
#include "base/utils/exceptions.h"
#include "base/script_options_parser/exception.h"
#include <sstream>


namespace ctpg_parser = ExecutionGraph::OptionsLineParser::CTPG;

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ScriptOptionLinesParserCTPG::ScriptOptionLinesParserCTPG()
: m_scriptCode()
, m_keywords(false)
, m_needParsing(true) {}

void ScriptOptionLinesParserCTPG::prepareScriptCode(const std::string & scriptCode) {
    m_scriptCode = scriptCode;
}

void ScriptOptionLinesParserCTPG::parseForScriptClass(std::function<void(const std::string &option)> callback) {
    try {
        parseForSingleOption(m_keywords.scriptClassKeyword(), callback);
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1623");
    }
}

void ScriptOptionLinesParserCTPG::parseForJvmOptions(std::function<void(const std::string &option)> callback) {
    try {
        parseForMultipleOption(m_keywords.jvmKeyword(), callback);
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1624");
    }
}

void ScriptOptionLinesParserCTPG::parseForExternalJars(std::function<void(const std::string &option)> callback) {
    try {
        parseForMultipleOption(m_keywords.jarKeyword(), callback);
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1625");
    }
}

void ScriptOptionLinesParserCTPG::extractImportScripts(SwigFactory & swigFactory) {

    try {
        parse();
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1626");
    }

    const auto optionIt = m_foundOptions.find(m_keywords.importKeyword());
    if (optionIt != m_foundOptions.end()) {
        CTPG::ScriptImporter scriptImporter(swigFactory, m_keywords);
        scriptImporter.importScript(m_scriptCode, m_foundOptions);
        //The imported scripts will change the location of the other options in m_foundOptions
        //Also there might be new JVM / External Jar options
        //=> We need to clear the option map and reset the parser.
        m_foundOptions.clear();
        m_needParsing = true;
    }
}

std::string && ScriptOptionLinesParserCTPG::getScriptCode() {
    try {
        parse();
    } catch(const ExecutionGraph::OptionParserException& ex) {
        Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1627");
    }
    //Remove all options from script code in reverse order
    struct option_location {
        size_t pos;
        size_t len;
    };
    struct comp {
        bool operator()(option_location a, option_location b) const {
            return a.pos > b.pos;
        }
    };
    std::set<option_location, comp> option_locations;
    for (const auto & option: m_foundOptions) {
        for (const auto & option_loc: option.second) {
            option_location loc = { .pos = option_loc.idx_in_source, .len = option_loc.size};
            option_locations.insert(loc);
        }
    }
    for (const auto option_loc: option_locations) {
        m_scriptCode.erase(option_loc.pos, option_loc.len);
    }
    return std::move(m_scriptCode);
}

void ScriptOptionLinesParserCTPG::parse() {
    if (m_needParsing) {
        if(!m_foundOptions.empty()) {
            throw std::logic_error(
                "F-UDF-CL-SL-JAVA-1620 Internal error. Parser result is not empty. "
                "Please open a bug ticket at https://github.com/exasol/script-languages-release/issues/new.");
        }
        try {
            ExecutionGraph::OptionsLineParser::CTPG::parseOptions(m_scriptCode, m_foundOptions);
        } catch(const ExecutionGraph::OptionParserException& ex) {
            Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1621");
        }

        m_needParsing = false;

        //Check for unknown options
        for (const auto & option: m_foundOptions) {
            if (m_keywords.jarKeyword() != option.first &&
                m_keywords.scriptClassKeyword() != option.first  &&
                m_keywords.importKeyword() != option.first  &&
                m_keywords.jvmKeyword() != option.first) {
                    std::stringstream ss;
                    ss << "F-UDF-CL-SL-JAVA-1622 " << "Unexpected option: " << option.first;
                    throw std::invalid_argument(ss.str());
                }
        }
    }
}

void ScriptOptionLinesParserCTPG::parseForSingleOption(const std::string key, std::function<void(const std::string &option)> callback) {
    parse();
    const auto optionIt = m_foundOptions.find(key);
    if (optionIt != m_foundOptions.end()) {
        if (optionIt->second.size() != 1) {
            std::stringstream ss;
            ss << "F-UDF-CL-SL-JAVA-1628 found " << optionIt->second.size() << " instances for script option key '" << key << "' but expected at most one." << std::endl;
            throw std::invalid_argument(ss.str());
        }
        callback(optionIt->second[0].value);
    }
}

void ScriptOptionLinesParserCTPG::parseForMultipleOption(const std::string key, std::function<void(const std::string &option)> callback) {
    parse();
    const auto optionIt = m_foundOptions.find(key);
    if (optionIt != m_foundOptions.end()) {
        for (const auto & option : optionIt->second) {
            callback(option.value);
        }
    }
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
