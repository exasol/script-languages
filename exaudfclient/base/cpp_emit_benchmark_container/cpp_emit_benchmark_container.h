#include "exaudflib/exaudflib.h"


using namespace SWIGVMContainers;
using namespace std;


class CppEmitBenchmarkVM: public SWIGVM {
    public:
        struct exception: SWIGVM::exception {
            exception(const char *reason): SWIGVM::exception(reason) { }
            virtual ~exception() throw() { }
        };
        CppEmitBenchmarkVM(bool checkOnly);
        virtual ~CppEmitBenchmarkVM() {};
        virtual void shutdown();
        virtual bool run();
        virtual const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
        virtual bool useZmqSocketLocks() {return true;}
    private:
        SWIGMetadata meta;
        SWIGTableIterator inp;
        SWIGResultHandler outp;
        void importScripts();
};
