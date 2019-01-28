#ifndef ENABLE_BENCHMARK_VM
#define ENABLE_BENCHMARK_VM
#endif

#include "benchmark_container.h"
#include "exaudflib.h"
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

BenchmarkVM::BenchmarkVM(bool checkOnly)
    : meta(), inp(), outp(&inp)

{
    if (meta.inputType()  != MULTIPLE
            || meta.outputType() != MULTIPLE)
    {
        throw SWIGVM::exception("BENCHMARK language container only support SET-EMITS UDFs");
    }
}

void BenchmarkVM::shutdown()
{
}

bool BenchmarkVM::run()
{
    try {

        try {
            bool hasNext = false;
            do {
                int count = 0;
                for (unsigned int col = 0; col<meta.inputColumnCount(); col++) {           
                    switch (meta.inputColumnType(col))
                    {
                        case DOUBLE:
                            {
                                auto v=inp.getDouble(col);
                                if(v==0.0){
                                    count++;
                                }
                            }
                            break;
                        case INT32:
                            {
                                auto v=inp.getInt32(col);
                                if(v==0){
                                    count++;
                                }
                            }
                            break;
                        case INT64:
                            {
                                auto v=inp.getInt64(col);
                                if(v==0){
                                    count++;
                                }
                            }
                            break;
                        case STRING:
                            {
                                auto v=new string(inp.getString(col));
                                if(v->empty()){
                                    count++;
                                }
                            }
                            break;                                                
                        default:
                            break;
                    }
                }
                hasNext = inp.next();
            } while (hasNext);
        } catch (std::exception& err) {
            lock_guard<mutex> lock(exception_msg_mtx);
            exception_msg = err.what();
        }

        try {
            outp.setString(0,"",0);
            outp.next();
            outp.setString(0,"",0);
            outp.flush();
        } catch (std::exception& err) {
            lock_guard<mutex> lock(exception_msg_mtx);
            exception_msg = err.what();
        }
        
        if (exception_msg.size() > 0) {
            throw SWIGVM::exception(exception_msg.c_str());
        }

    } catch (std::exception& ex) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = ex.what();
    }
    return true;  // done
}

const char* BenchmarkVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args)
{
    throw SWIGVM::exception("singleCall not supported for the STREAMING language container");
}
