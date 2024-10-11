#include "base/javacontainer/script_options/extractor.h"
#include "base/javacontainer/script_options/parser.h"

#include "base/utils/debug_message.h"
#include <iostream>

#define EXTR_DBG_FUNC_CALL(f) DBG_FUNC_CALL(std::cerr, f)

namespace SWIGVMContainers {

namespace JavaScriptOptions {

Extractor::Extractor(ScriptOptionsParser & parser,
                     SwigFactory& swigFactory)
: m_parser(parser)
, m_swigFactory(swigFactory) {}

void Extractor::extract(std::string & scriptCode) {
    m_parser.prepareScriptCode(scriptCode);
    EXTR_DBG_FUNC_CALL(m_parser.parseForScriptClass( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertScriptClassName(value)); // To be called before scripts are imported. Otherwise, the script classname from an imported script could be used
        }));
    EXTR_DBG_FUNC_CALL(m_parser.extractImportScripts(m_swigFactory));
    EXTR_DBG_FUNC_CALL(m_parser.parseForJvmOptions( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertJvmOption(value));
        }));
    EXTR_DBG_FUNC_CALL(m_parser.parseForExternalJars( [&](const std::string& value){
            EXTR_DBG_FUNC_CALL(m_converter.convertExternalJar(value));
        }));
    scriptCode = std::move(m_parser.getScriptCode());
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
