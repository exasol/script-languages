#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/parserlegacy.h"
#include "base/exaudflib/swig/swig_meta_data.h"
#include <openssl/md5.h>
#include <string.h>



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


ScriptOptionsConverter::ScriptOptionsConverter(std::function<void(const std::string&)> throwException,
                                               std::vector<std::string>& jvmOptions):
m_scriptOptionsParser(std::make_unique<ScriptOptionLinesParserLegacy>()),
m_throwException(throwException),
m_jvmOptions(jvmOptions),
m_jarPaths()
{}

void ScriptOptionsConverter::getExternalJarPaths(std::string & src_scriptCode) {
    std::vector<std::string> jarPaths;
    m_scriptOptionsParser->findExternalJarPaths(src_scriptCode, jarPaths,
                                                [&](const std::string& msg){m_throwException("F-UDF-CL-SL-JAVA-1600" + msg);});
    for (const std::string& jarPath : jarPaths) {
        for (size_t start = 0, delim = 0; ; start = delim + 1) {
            delim = jarPath.find(":", start);
            if (delim != std::string::npos) {
                std::string jar = jarPath.substr(start, delim - start);
                if (m_jarPaths.find(jar) == m_jarPaths.end())
                    m_jarPaths.insert(jar);
            }
            else {
                std::string jar = jarPath.substr(start);
                if (m_jarPaths.find(jar) == m_jarPaths.end())
                    m_jarPaths.insert(jar);
                break;
            }
        }
    }
}

void ScriptOptionsConverter::getScriptClassName(std::string & src_scriptCode) {
    std::string scriptClass;

    m_scriptOptionsParser->getScriptClassName(src_scriptCode, scriptClass,
                              [&](const std::string& msg){m_throwException("F-UDF-CL-SL-JAVA-1601: " + msg);});

    if (scriptClass != "") {
        m_jvmOptions.push_back("-Dexasol.scriptclass=" + scriptClass);
    }
}

void ScriptOptionsConverter::getExternalJvmOptions(std::string & src_scriptCode) {
    std::vector<std::string> jvmOptions;
    const std::string whitespace = " \t\f\v";
    m_scriptOptionsParser->getExternalJvmOptions(src_scriptCode, jvmOptions,
                                                 [&](const std::string& msg){m_throwException("F-UDF-CL-SL-JAVA-1602" + msg);});

    for (const std::string& jvmOption:  jvmOptions) {
        for (size_t start = 0, delim = 0; ; start = delim + 1) {
            start = jvmOption.find_first_not_of(whitespace, start);
            if (start == std::string::npos)
                break;
            delim = jvmOption.find_first_of(whitespace, start);
            if (delim != std::string::npos) {
                m_jvmOptions.push_back(jvmOption.substr(start, delim - start));
            }
            else {
                m_jvmOptions.push_back(jvmOption.substr(start));
                break;
            }
        }
    }
}

void ScriptOptionsConverter::convertImportScripts(std::string & src_scriptCode) {
    SWIGMetadata *meta = NULL;
    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports its self
    m_importedScriptChecksums.insert(scriptToMd5(src_scriptCode.c_str()));
    while (true) {
        std::pair<std::string, size_t> nextImportStatement;
        m_scriptOptionsParser->getNextImportScript(src_scriptCode, nextImportStatement,
                                                     [&](const std::string& msg){m_throwException("F-UDF-CL-SL-JAVA-1604" + msg);});
        if (nextImportStatement.first == "")
            break;
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
    if (meta)
        delete meta;
}

const std::set<std::string> & ScriptOptionsConverter::getJarPaths() {
    return m_jarPaths;
}

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
