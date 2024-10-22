#ifndef SCRIPTOPTIONLINEPARSER_H
#define SCRIPTOPTIONLINEPARSER_H 1

#include <string>
#include <vector>
#include <functional>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser {
    virtual ~ScriptOptionsParser() {};
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
    virtual void parseForScriptClass(std::function<void(const std::string &option)> callback) = 0;
    /*
    Searches for JVM options.
    If an option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode".
    */
    virtual void parseForJvmOptions(std::function<void(const std::string &option)> callback) = 0;

    /*
    Searches for External Jar.
    If an option is found, the function removes the option from "scriptCode" and calls "callback" with the option value and position
    within "scriptCode".
    */
    virtual void parseForExternalJars(std::function<void(const std::string &option)> callback) = 0;

    /*
     Searches for the "%import" options and embeds the respective imported script code at the same location as
     the option in the script code.
    */
    virtual void extractImportScripts() = 0;

    /*
     Returns the (eventually modified) script code.
    */
    virtual std::string && getScriptCode() = 0;

    /*
     Returns true if parser supports escape sequences.
     */
    virtual bool supportsEscapeSequences() = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers


#endif //SCRIPTOPTIONLINEPARSER_H