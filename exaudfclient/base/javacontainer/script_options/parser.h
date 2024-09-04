#ifndef SCRIPTOPTIONLINEPARSER_H
#define SCRIPTOPTIONLINEPARSER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser {

    virtual void findExternalJarPaths(std::string & src_scriptCode,
                                      std::vector<std::string>& jarPaths,
                                      std::function<void(const std::string&)> throwException) = 0;

    virtual void getScriptClassName(std::string & src_scriptCode, std::string &scriptClassName,
                                    std::function<void(const std::string&)> throwException) = 0;

    virtual void getNextImportScript(std::string & src_scriptCode,
                                     std::pair<std::string, size_t> & result,
                                     std::function<void(const std::string&)> throwException) = 0;

    virtual void getExternalJvmOptions(std::string & src_scriptCode,
                                       std::vector<std::string>& jvmOptions,
                                       std::function<void(const std::string&)> throwException) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H