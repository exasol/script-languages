#include "base/javacontainer/script_options/extractor_impl.h"
#include "base/utils/debug_message.h"
#include <iostream>

#define EXTR_DBG_FUNC_CALL(f) DBG_FUNC_CALL(std::cerr, f)

namespace SWIGVMContainers {

namespace JavaScriptOptions {

template<typename TParser, typename TConverter>
ExtractorImpl<TParser, TConverter>::ExtractorImpl(std::unique_ptr<SwigFactory> swigFactory)
: m_parser(std::move(swigFactory))
, m_converter() {}

template<typename TParser, typename TConverter>
inline void ExtractorImpl<TParser, TConverter>::iterateJarPaths(std::function<void(const std::string &option)> callback) const {
    m_converter.iterateJarPaths(callback);
}

template<typename TParser, typename TConverter>
inline std::vector<std::string>&& ExtractorImpl<TParser, TConverter>::moveJvmOptions() {
    return std::move(m_converter.moveJvmOptions());
}

template<typename TParser, typename TConverter>
void ExtractorImpl<TParser, TConverter>::extract(std::string & scriptCode) {
    m_parser.prepareScriptCode(scriptCode);
    EXTR_DBG_FUNC_CALL(m_parser.parseForScriptClass( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertScriptClassName(value)); // To be called before scripts are imported. Otherwise, the script classname from an imported script could be used
        }));
    EXTR_DBG_FUNC_CALL(m_parser.extractImportScripts());
    EXTR_DBG_FUNC_CALL(m_parser.parseForJvmOptions( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertJvmOption(value));
        }));

    EXTR_DBG_FUNC_CALL(m_parser.parseForExternalJars( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertExternalJar(value));
        }));

    scriptCode = std::move(m_parser.getScriptCode());
}


// Explict class template instantiations
template class ExtractorImpl<ScriptOptionLinesParserLegacy, ConverterLegacy>;
template class ExtractorImpl<ScriptOptionLinesParserCTPG, ConverterV2>;


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
