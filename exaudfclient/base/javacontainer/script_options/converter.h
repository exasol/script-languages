#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/parser.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Converter {

    public:
        Converter();
    
        void convertExternalJar(const std::string & value);

        void convertScriptClassName(const std::string & value);

        void convertJvmOption(const std::string & value);

        const std::set<std::string> & getJarPaths() const {
            return m_jarPaths;
        }

        const std::vector<std::string>& getJvmOptions() const {
            return m_jvmOptions;
        }

        const std::string& getScriptClassName() const {
            return m_scriptClassName;
        }

    private:

        std::vector<std::string> m_jvmOptions;
        
        std::set<std::string> m_jarPaths;

        std::string m_scriptClassName;
};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H