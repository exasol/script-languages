#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>


#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

class SWIGMetadata;

namespace JavaScriptOptions {

class ScriptOptionsParser;

class Extractor {

    public:
        Extractor(ScriptOptionsParser & parser,
                  std::function<void(const std::string&)> throwException);

        const std::set<std::string> & getJarPaths() const {
            return m_converter.getJarPaths();
        }

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_converter.moveJvmOptions());
        }

        void extract(std::string & scriptCode);

    private:
        std::function<void(const std::string&)> m_throwException;

        Converter m_converter;
        ScriptOptionsParser & m_parser;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H