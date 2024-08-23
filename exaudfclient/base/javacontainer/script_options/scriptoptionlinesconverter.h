#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <functional>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/scriptoptionlinesparser.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ScriptOptionsConverter {

    public:
        ScriptOptionsConverter(std::function<void(const std::string &msg)> throwException,
                               std::vector<std::string>& jvmOptions);

    
        void getExternalJarPaths(std::string & src_scriptCode);

        void getScriptClassName(std::string & src_scriptCode);

        void convertImportScripts(std::string & src_scriptCode);

        void getExternalJvmOptions(std::string & src_scriptCode);

        const std::set<std::string> & getJarPaths();
    private:
    
        std::unique_ptr<ScriptOptionsParser> m_scriptOptionsParser;
        
        std::function<void(const std::string &msg)> m_throwException;

        std::vector<std::string>& m_jvmOptions;
        
        std::set<std::string> m_jarPaths;

        std::set<std::vector<unsigned char> > m_importedScriptChecksums;
};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H