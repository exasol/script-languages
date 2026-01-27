#ifndef SCRIPTOPTIONLINEPEXTRACTORIMPL_H
#define SCRIPTOPTIONLINEPEXTRACTORIMPL_H 1

#include <string>
#include <vector>
#include <set>


#include "javacontainer/script_options/extractor.h"
#include "javacontainer/script_options/converter_legacy.h"
#include "javacontainer/script_options/converter_v2.h"
#include "javacontainer/script_options/parser_ctpg.h"
#include "javacontainer/script_options/parser_legacy.h"
#include "swig_factory/swig_factory.h"

namespace SWIGVMContainers {

namespace JavaScriptOptions {

/**
 * Concrete implementation for the Extractor class.
 * Given template parameter TParser and TConverter define concrete behavior.
 */
template<typename TParser, typename TConverter>
class ExtractorImpl : public Extractor {

    public:

        ExtractorImpl(std::unique_ptr<SwigFactory> swigFactory);

        virtual void iterateJarPaths(tJarIteratorCallback callback) const override;
        std::vector<std::string>&& moveJvmOptions() override;

        void extract(std::string & scriptCode);

    private:
        TParser m_parser;
        TConverter m_converter;
};

typedef ExtractorImpl<ScriptOptionLinesParserLegacy, ConverterLegacy> tExtractorLegacy;
typedef ExtractorImpl<ScriptOptionLinesParserCTPG, ConverterV2> tExtractorV2;

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTORIMPL_H