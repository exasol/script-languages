
using namespace SWIGVMContainers;
using namespace std;


class StreamingVM: public SWIGVM {
    public:
        struct exception: SWIGVM::exception {
            exception(const char *reason): SWIGVM::exception(reason) { }
            virtual ~exception() throw() { }
        };
        StreamingVM(bool checkOnly);
        virtual ~StreamingVM() {};
        virtual void shutdown();
        virtual bool run();
        virtual const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
        virtual bool useZmqSocketLocks() {return true;}
    private:
        SWIGMetadata meta;
        SWIGTableIterator inp;
        SWIGResultHandler outp;
        void importScripts();
        map<SWIGVM_datatype_e, std::function<void(ostream&str, unsigned int col)> > csvPrinters;
        map<SWIGVM_datatype_e, std::function<void(string& str, unsigned int col)> > csvReaders;
        void inputToCSV(ostream&str);
        bool CSVToOutput(istream&str);
        string readBuffer;
        // returns true if a an entry could be read
        bool readCSVValue(istream&str, unsigned int column);
};
