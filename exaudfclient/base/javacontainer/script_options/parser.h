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
    virtual void prepareScriptCode(const std::string & scriptCode) = 0;
    /*
    Searches for script class option.
    If the option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode".
    */
    virtual void parseForScriptClass(std::function<void(const std::string &option)> callback,
                                     std::function<void(const std::string&)> throwException) = 0;
    /*
    Searches for JVM options.
    If an option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode".
    */
    virtual void parseForJvmOptions(std::function<void(const std::string &option)> callback,
                                     std::function<void(const std::string&)> throwException) = 0;

    /*
    Searches for External Jar.
    If an option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode".
    */
    virtual void parseForExternalJars(std::function<void(const std::string &option)> callback,
                                      std::function<void(const std::string&)> throwException) = 0;

    virtual void extractImportScripts(std::function<void(const std::string&)> throwException) = 0;

    /*
     Returns the (eventually modified) script code.
    */
    virtual std::string && getScriptCode() = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H