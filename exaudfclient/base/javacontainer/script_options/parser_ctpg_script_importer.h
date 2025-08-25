#ifndef SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H
#define SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H 1

#include "base/javacontainer/script_options/checksum.h"
#include "base/javacontainer/script_options/keywords.h"
#include "base/exaudflib/swig/swig_meta_data.h"
#include "base/script_options_parser/ctpg/script_option_lines_ctpg.h"
#include <memory>
#include <unordered_set>

namespace SWIGVMContainers {

    struct SwigFactory;

namespace JavaScriptOptions {

namespace CTPG {



class ScriptImporter {

    public:
        ScriptImporter(SwigFactory & swigFactory, Keywords & keywords);

        void importScript(std::string & scriptCode, ExecutionGraph::OptionsLineParser::CTPG::options_map_t & options);

    private:
         struct CollectedScript {
            CollectedScript(CollectedScript&&) = default;
            std::string script;
            size_t origPos;
            size_t origLen;
        };

        typedef ExecutionGraph::OptionsLineParser::CTPG::options_map_t::mapped_type OptionValues_t;

        void importScript(std::string & scriptCode,
                            ExecutionGraph::OptionsLineParser::CTPG::options_map_t & options,
                            const size_t recursionDepth);
         const char* findImportScript(const std::string & scriptKey);

         void collectImportScripts(const OptionValues_t & option_values,
                                   const size_t recursionDepth,
                                   std::vector<CollectedScript> &result);

         void replaceImportScripts(std::string & scriptCode,
                                   const std::vector<CollectedScript> &collectedImportScripts);

        std::unordered_set<std::string> m_importedSetOfScripts;
        SwigFactory & m_swigFactory;
        std::unique_ptr<SWIGMetadataIf> m_metaData;
        Keywords & m_keywords;
        //The empirical maximal value for recursion depth is ~18000. So we add a little bit extra to have some buffer.
        const size_t cMaxRecursionDepth = 10000U;
};

} //namespace CTPG

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H
