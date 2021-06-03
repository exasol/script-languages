#ifndef ENABLE_BENCHMARK_VM
#define ENABLE_BENCHMARK_VM
#endif

#include "debug_message.h"
#include "cpp_emit_benchmark_container.h"
#include "exaudflib/exaudflib.h"
#include <iostream>
#include <string>
#include <functional>
#include <map>
#include <sstream>
#include <unistd.h>
#include <ext/stdio_filebuf.h>
#include <fstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <mutex>
#include <type_traits>
#include <string>
using namespace SWIGVMContainers;
using namespace std;

CppEmitBenchmarkVM::CppEmitBenchmarkVM(bool checkOnly)
    : meta(), inp(), outp(&inp)

{
    if (meta.inputType()  != MULTIPLE
            || meta.outputType() != MULTIPLE)
    {
        throw SWIGVM::exception("BENCHMARK language container only support SET-EMITS UDFs");
    }
}

void CppEmitBenchmarkVM::shutdown()
{
}

bool CppEmitBenchmarkVM::run()
{
    DBG_FUNC_BEGIN(cerr);
    cerr << "Input Columns:" << endl;
    for (unsigned int col = 0; col<meta.inputColumnCount(); col++) {
        cerr << "Column " << col << ": " << meta.inputColumnName(col) << " " << meta.inputColumnTypeName(col) << endl;
    }
        try {
            uint64_t count = 0; 
            for (unsigned int col = 0; col<meta.inputColumnCount(); col++) {           
                switch (meta.inputColumnType(col))
                {
                    case INT32:
                        {
                            auto v=inp.getInt32(col);
                            if(v>0){
                                count=static_cast<uint64_t>(v);
                            }
                        }
                        break;
                    case INT64:
                        {
                            auto v=inp.getInt64(col);
                            if(v>0){
                                count=static_cast<uint64_t>(v);
                            }
                        }
                        break;
                    default:
                        throw SWIGVM::exception("Wrong Input Type");
                }
            }

            string str1 = string("999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999");
            for(size_t i=0;i<count;i++){
              outp.setString(0,str1.c_str(),strlen(str1.c_str()));
              outp.next();
            }
            outp.flush();

        } catch (std::exception& err) {
            lock_guard<mutex> lock(exception_msg_mtx);
            exception_msg = err.what();
            cerr << "Exception BenchmarkVM::run " << exception_msg << endl;
            
        
            if (exception_msg.size() > 0) {
                throw SWIGVM::exception(exception_msg.c_str());
            }
        }


    return true;  // done
}

const char* CppEmitBenchmarkVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args)
{
    throw SWIGVM::exception("singleCall not supported for the STREAMING language container");
}
