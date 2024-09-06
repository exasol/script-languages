#include "base/javacontainer/script_options/extractor.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/exaudflib/swig/swig_meta_data.h"
#include <openssl/md5.h>


namespace SWIGVMContainers {

namespace JavaScriptOptions {


inline std::vector<unsigned char> scriptToMd5(const char *script) {
    MD5_CTX ctx;
    unsigned char md5[MD5_DIGEST_LENGTH];
    MD5_Init(&ctx);
    MD5_Update(&ctx, script, strlen(script));
    MD5_Final(md5, &ctx);
    return std::vector<unsigned char>(md5, md5 + sizeof(md5));
}

Extractor::Extractor(const std::string scriptCode, std::function<void(const std::string&)> throwException)
: m_modifiedCode(scriptCode)
, m_throwException(throwException)
, m_jarKeyword("%jar")
, m_scriptClassKeyword("%scriptclass")
, m_importKeyword("%import")
, m_jvmOptionKeyword("%jvmoption") {}


ScriptOptionsParser* Extractor::makeParser() {
    return new ScriptOptionLinesParserLegacy();
}

void extractImportScripts(ScriptOptionsParser *parser) {
    SWIGMetadata *meta = NULL;
    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports its self
    std::set<std::vector<unsigned char> > importedScriptChecksums;
    importedScriptChecksums.insert(scriptToMd5(m_modifiedCode.c_str()));
    extractImportScript()
    if (meta)
        delete meta;
}

void extractImportScript(SWIGMetadata** metaData, std::string & scriptCode, ScriptOptionsParser *parser) {
    while (true) {
        std::string importScript;
        size_t importScriptPos;
        parser->parseForSingleOption(m_importKeyword,
                                        [](const std::string& value, size_t pos){importScript=value; importScriptPos=pos;}
                                        [&](const std::string& msg){m_throwException("F-UDF-CL-SL-JAVA-1604" + msg);});
        if importScriptPos
        if (!meta) {
            meta = new SWIGMetadata();
            if (!meta)
                m_throwException("F-UDF-CL-SL-JAVA-1603: Failure while importing scripts");
        }
        const char *scriptCode = meta->moduleContent(nextImportStatement.first.c_str());
        const char *exception = meta->checkException();
        if (exception)
            m_throwException("F-UDF-CL-SL-JAVA-1605: "+std::string(exception));
        if (m_importedScriptChecksums.insert(scriptToMd5(scriptCode)).second) {
            // Script has not been imported yet
            // If this imported script contains %import statements
            // they will be resolved in this while loop.
            src_scriptCode.insert(nextImportStatement.second, scriptCode);
        }
    }
    }
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
