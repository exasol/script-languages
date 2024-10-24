#ifndef SCRIPTOPTIONLINEPARSERCONVERTERLEGACY_H
#define SCRIPTOPTIONLINEPARSERCONVERTERLEGACY_H 1

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

class ConverterLegacy : public Converter {

    public:
        ConverterLegacy();
    
        void convertExternalJar(const std::string & value);

        void iterateJarPaths(std::function<void(const std::string &option)> callback) const override;

    private:

        std::set<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H