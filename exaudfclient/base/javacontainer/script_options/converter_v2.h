#ifndef SCRIPTOPTIONLINEPARSERCONVERTERV2_H
#define SCRIPTOPTIONLINEPARSERCONVERTERV2_H 1

#include <string>
#include <vector>
#include <memory>

#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

/**
 * This class is a specialization for the generic converter class.
 * It implements conversion of the jar option according to the requirements in the new
 * parser implementation.
 */
class ConverterV2 : public Converter {

    public:
        ConverterV2();
    
        void convertExternalJar(const std::string & value) override;

        void convertJvmOption(const std::string & value)  override;

        void iterateJarPaths(tJarIteratorCallback callback) const override;

    private:
        
        std::vector<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H