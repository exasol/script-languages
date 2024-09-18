#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>


#include "base/javacontainer/script_options/converter.h"
#include "base/javacontainer/script_options/parser.h"


namespace SWIGVMContainers {

class SWIGMetadata;

namespace JavaScriptOptions {

class ParserFactory;

class Extractor {

    public:
        Extractor(ParserFactory & parserFactory,
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
        ParserFactory & m_parserFactory;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H