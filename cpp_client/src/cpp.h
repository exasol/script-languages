#ifndef CPP_H
#define CPP_H

#include "swigcontainers_ext.h"
#include <vector>
#include <string>
#include <scriptDTO.h>
#include <set>

namespace SWIGVMContainers {
class CPPVM: public SWIGVM {

    typedef void (*RUN_FUNC)(const SWIGVMContainers::SWIGMetadata&, SWIGVMContainers::SWIGTableIterator&, SWIGVMContainers::SWIGResultHandler&);

    typedef std::string (*DEFAULT_OUTPUT_COLUMNS_FUNC)(const SWIGVMContainers::SWIGMetadata&);
    typedef std::string (*ADAPTER_CALL_FUNC)(const SWIGVMContainers::SWIGMetadata&, const std::string input);
    typedef std::string (*IMPORT_ALIAS_FUNC)(const SWIGVMContainers::SWIGMetadata&, const ExecutionGraph::ImportSpecification& importSpecification);
    typedef void (*CLEANUP_FUNC)();

    RUN_FUNC run_cpp=NULL;

    public:
        struct exception: SWIGVM::exception {
            exception(const char *reason): SWIGVM::exception(reason) { }
            virtual ~exception() throw() { }
        };

        CPPVM(bool checkOnly);
        virtual ~CPPVM() {}
        virtual void shutdown();
        virtual bool run();
        virtual std::string singleCall(single_call_function_id fn, const ExecutionGraph::ScriptDTO& args);
private:

        void importScripts();
        std::vector<unsigned char> scriptToMd5(const char *script);
        std::string getOptionLine(std::string scriptCode, const std::string option, const std::string whitespace, const std::string lineEnd, size_t& pos);

       SWIGMetadata meta;
       
       std::set< std::vector<unsigned char> > m_importedScriptChecksums;
       
       std::string m_script_code;
       void* handle;
};
}
#endif // CPP_H
