#ifndef SCRIPTOPTIONLINEPARSERCONVERTERV2_H
#define SCRIPTOPTIONLINEPARSERCONVERTERV2_H 1

#include <string>
#include <vector>
#include <memory>

#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ConverterV2 : public Converter {

    public:
        ConverterV2();
    
        void convertExternalJar(const std::string & value);

        void iterateJarPaths(tJarIteratorCallback callback) const override;

    private:
        
        std::vector<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H