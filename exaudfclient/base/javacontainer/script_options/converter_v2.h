#ifndef SCRIPTOPTIONLINEPARSERCONVERTERV2_H
#define SCRIPTOPTIONLINEPARSERCONVERTERV2_H 1

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ConverterV2 : public Converter {

    public:
        ConverterV2();
    
        void convertExternalJar(const std::string & value);

        const std::set<std::string> & getJarPaths() const {
            return m_jarPaths;
        }

    private:
        
        std::set<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H