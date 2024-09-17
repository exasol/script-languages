#ifndef SCRIPTOPTIONLINEPARSER_H
#define SCRIPTOPTIONLINEPARSER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser {
    /*
    Passes the script code for parsing to the parser. The parser might modify the script code, because it will remove
    known options.
    */
    void prepareScriptCode(const std::string & scriptCode);
    /*
    Searches for one specific option, identified by parameter "key", in parameter "scriptCode".
    If the option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode.
    */
    virtual void parseForSingleOption(const std::string key,
                                        std::function<void(const std::string &option, size_t pos)> callback,
                                        std::function<void(const std::string&)> throwException) = 0;
    /*
    Searches for multiple options, identified by parameter "key", in script code which was given
    to the parser previously via function "prepareScriptCode".
    If an option an option is found, the function removes the option from the script code
    and calls "callback" with the option value and position within "scriptCode.
    The order order of options is is not defined and can be different for different parser implementations.
    */
    virtual void parseForMultipleOptions(const std::string key,
                                            std::function<void(const std::string &option, size_t pos)> callback,
                                            std::function<void(const std::string&)> throwException) = 0;
    /*
     Returns the (eventually modified) script code.
    */
    std::string getScriptCode() const;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H