#ifndef SCRIPTOPTIONLINEPARSER_H
#define SCRIPTOPTIONLINEPARSER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser {
    virtual void parseForSingleOption(const std::string key,
                                        std::function<void(const std::string &option, size_t pos)> callback,
                                        std::function<void(const std::string&)> throwException) = 0;
    virtual void parseForMultipleOptions(const std::string key,
                                            std::function<void(const std::string &option, size_t pos)> callback,
                                            std::function<void(const std::string&)> throwException) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H