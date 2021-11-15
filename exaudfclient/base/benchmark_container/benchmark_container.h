
using namespace SWIGVMContainers;
using namespace std;


class BenchmarkVM: public SWIGVM {
    public:
        struct exception: SWIGVM::exception {
            exception(const char *reason): SWIGVM::exception(reason) { }
            virtual ~exception() throw() { }
        };
        BenchmarkVM(bool checkOnly);
        virtual ~BenchmarkVM() {};
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