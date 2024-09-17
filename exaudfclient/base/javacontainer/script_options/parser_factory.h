#ifndef SCRIPTOPTIONLINEPARSER_FACTORY_H
#define SCRIPTOPTIONLINEPARSER_FACTORY_H 1

#include <memory>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct ScriptOptionsParser;

struct ParserFactory {
    virtual std::unique_ptr<ScriptOptionsParser> makeParser(std::string & scriptCode) = 0;
};

struct ParserFactoryLegacy : public ParserFactory {
    std::unique_ptr<ScriptOptionsParser> makeParser(std::string & scriptCode) override;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
#endif