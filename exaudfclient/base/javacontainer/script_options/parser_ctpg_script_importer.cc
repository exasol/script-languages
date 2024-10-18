#include "base/javacontainer/script_options/parser_ctpg_script_importer.h"
#include "base/swig_factory/swig_factory.h"
#include "base/utils/debug_message.h"
#include "base/utils/exceptions.h"
#include "base/script_options_parser/exception.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace ctpg_parser = ExecutionGraph::OptionsLineParser::CTPG;

namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace CTPG {

ScriptImporter::ScriptImporter(SwigFactory & swigFactory, Keywords & keywords)
: m_importedScriptChecksums()
, m_swigFactory(swigFactory)
, m_metaData()
, m_keywords(keywords) {}

void ScriptImporter::importScript(std::string & scriptCode,
                                    ctpg_parser::options_map_t & options) {
    importScript(scriptCode, options, 0);
}

void ScriptImporter::collectImportScripts(const ScriptImporter::OptionValues_t & option_values,
                                          const size_t recursionDepth,
                                          std::vector<CollectedScript> &result) {
    for (const auto & option: option_values) {
        const char *importScriptCode = findImportScript(option.value);
        std::string importScriptCodeStr;
        if (m_importedScriptChecksums.addScript(importScriptCode) ) {
            // Script has not been imported yet
            // If this imported script contains %import statements
            // they will be resolved in the next recursion.
            ctpg_parser::options_map_t newOptions;
            try {
                ExecutionGraph::OptionsLineParser::CTPG::parseOptions(importScriptCode, newOptions);
            } catch(const ExecutionGraph::OptionParserException & ex) {
                Utils::rethrow(ex, "F-UDF-CL-SL-JAVA-1630");
            }
            importScriptCodeStr.assign(importScriptCode);
            importScript(importScriptCodeStr, newOptions, recursionDepth + 1);
        }
        CollectedScript replacedScript = {.script = std::move(importScriptCodeStr), .origPos = option.idx_in_source, .origLen = option.size };
        result.push_back(std::move(replacedScript));
    }
}

void ScriptImporter::replaceImportScripts(std::string & scriptCode,
                                          const std::vector<CollectedScript> &collectedImportScripts) {
    //Replace the imported script bodies from end to start.
    //Doing it in forward order would invalidate the offsets of later import scripts.
    for (auto optionIt = collectedImportScripts.crbegin(); optionIt != collectedImportScripts.crend(); optionIt++) {
        scriptCode.replace(optionIt->origPos, optionIt->origLen, optionIt->script);
    }
}


void ScriptImporter::importScript(std::string & scriptCode,
                                    ctpg_parser::options_map_t & options,
                                    const size_t recursionDepth) {
    const auto optionIt = options.find(std::string(m_keywords.importKeyword()));

    if (recursionDepth >= cMaxRecursionDepth) {
        throw std::runtime_error("F-UDF-CL-SL-JAVA-1633: Maximal recursion depth for importing scripts reached.");
    }
    if (optionIt != options.end()) {
        m_importedScriptChecksums.addScript(scriptCode.c_str());
        //Sort options from first in script to last in script
        std::sort(optionIt->second.begin(), optionIt->second.end(),
                    [](const ctpg_parser::ScriptOption& first, const ctpg_parser::ScriptOption& second)
                                  {
                                      return first.idx_in_source < second.idx_in_source;
                                  });
        std::vector<CollectedScript> collectedScript;
        collectedScript.reserve(optionIt->second.size());
        //In order to continue compatibility with legacy implementation we must collect import scripts in forward direction
        //but then replace in reverse direction (in order to keep consistency of positions)
        collectImportScripts(optionIt->second, recursionDepth, collectedScript);
        //Now replace the imported script bodies
        replaceImportScripts(scriptCode, collectedScript);
    }
}

const char* ScriptImporter::findImportScript(const std::string & scriptKey) {
    if (!m_metaData) {
        m_metaData.reset(m_swigFactory.makeSwigMetadata());
        if (!m_metaData) {
            throw std::runtime_error("F-UDF-CL-SL-JAVA-1631: Failure while importing scripts");
        }
    }
    const char *importScriptCode = m_metaData->moduleContent(scriptKey.c_str());
    const char *exception = m_metaData->checkException();
    if (exception) {
        throw std::runtime_error("F-UDF-CL-SL-JAVA-1632: " + std::string(exception));
    }
    return importScriptCode;
}

} //namespace CTPG

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
