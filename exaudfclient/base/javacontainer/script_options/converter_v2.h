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

        void iterateJarPaths(std::function<void(const std::string &option)> callback) const override {
            for (const auto & jar : m_jarPaths) {
                callback(jar);
            }
        }

    private:
        
        std::vector<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H