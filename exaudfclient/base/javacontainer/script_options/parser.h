#ifndef SCRIPTOPTIONLINEPARSER_H
#define SCRIPTOPTIONLINEPARSER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser {
    /*
    Searches for one specific option, identified by parameter "key", in parameter "scriptCode".
    If the option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode.
    */
    virtual void parseForSingleOption(std::string & scriptCode, const std::string key,
                                        std::function<void(const std::string &option, size_t pos)> callback,
                                        std::function<void(const std::string&)> throwException) = 0;
    /*
    Searches for multiple options, identified by parameter "key", in parameter "scriptCode".
    If an option an option is found, the function removes the option from "scriptCode"
    and calls "callback" with the option value and position within "scriptCode.
    The order order of options is is not defined and can be different for different parser implementations.
    */
    virtual void parseForMultipleOptions(std::string & scriptCode, const std::string key,
                                            std::function<void(const std::string &option, size_t pos)> callback,
                                            std::function<void(const std::string&)> throwException) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H