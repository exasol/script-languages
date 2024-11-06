#ifndef SCRIPTOPTIONLINEPARSERCONVERTERLEGACY_H
#define SCRIPTOPTIONLINEPARSERCONVERTERLEGACY_H 1

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "base/javacontainer/script_options/converter.h"



namespace SWIGVMContainers {

namespace JavaScriptOptions {

/**
 * This class is a specialization for the generic converter class.
 * It implements conversion of the jar option according to the requirements in the old
 * parser implementation.
 */
class ConverterLegacy : public Converter {

    public:
        ConverterLegacy();
    
        void convertExternalJar(const std::string & value) override;

        void convertJvmOption(const std::string & value)  override;

        void iterateJarPaths(tJarIteratorCallback callback) const override;

    private:

        std::set<std::string> m_jarPaths;

};




} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H