#ifndef SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H
#define SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H 1

#include "base/javacontainer/script_options/checksum.h"
#include "base/javacontainer/script_options/keywords.h"
#include "base/exaudflib/swig/swig_meta_data.h"
#include "base/script_options_parser/ctpg/script_option_lines_ctpg.h"
#include <memory>


namespace SWIGVMContainers {

    struct SwigFactory;

namespace JavaScriptOptions {

namespace CTPG {



class ScriptImporter {

    public:
        ScriptImporter(SwigFactory & swigFactory, Keywords & keywords);

        void importScript(std::string & scriptCode, ExecutionGraph::OptionsLineParser::CTPG::options_map_t & options);

    private:
         struct ReplacedScripts {
            ReplacedScripts(ReplacedScripts&&) = default;
            std::string script;
            size_t origPos;
            size_t origLen;
        };

    private:
        void importScript(std::string & scriptCode,
                            ExecutionGraph::OptionsLineParser::CTPG::options_map_t & options,
                            const size_t recursionDepth);
         const char* findImportScript(const std::string & scriptKey);

         void replaceScripts(const ExecutionGraph::OptionsLineParser::CTPG::options_map_t::mapped_type & option_values,
                             const size_t recursionDepth,
                             std::vector<ReplacedScripts> &result);

    private:
        Checksum m_importedScriptChecksums;
        SwigFactory & m_swigFactory;
        std::unique_ptr<SWIGMetadataIf> m_metaData;
        Keywords & m_keywords;
        //The empirical maximal value for recursion depth is ~18000. So we choose 10000 to have a certain buffer.
        const size_t cMaxRecursionDepth = 20000;
};

} //namespace CTPG

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H
