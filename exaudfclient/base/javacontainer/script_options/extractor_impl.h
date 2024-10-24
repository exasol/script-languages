#ifndef SCRIPTOPTIONLINEPEXTRACTORIMPL_H
#define SCRIPTOPTIONLINEPEXTRACTORIMPL_H 1

#include <string>
#include <vector>
#include <set>


#include "base/javacontainer/script_options/extractor.h"
#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/parser_ctpg.h"
#include "base/javacontainer/script_options/parser_legacy.h"
#include "base/swig_factory/swig_factory.h"

namespace SWIGVMContainers {

namespace JavaScriptOptions {

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