#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/parser.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Converter {

    public:
        Converter();
    
        void convertExternalJar(const std::string & value);

        void convertExternalJarWithEscapeSequences(const std::string & value);

        void convertScriptClassName(const std::string & value);

        void convertJvmOption(const std::string & value);

        const std::set<std::string> & getJarPaths() const {
            return m_jarPaths;
        }

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_jvmOptions);
        }

    private:

        std::vector<std::string> m_jvmOptions;
        
        std::set<std::string> m_jarPaths;

        const std::string m_whitespace;
};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H