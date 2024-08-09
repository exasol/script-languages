#ifndef SWIGVM_H
#define SWIGVM_H

#include <exception>
#include <string>
#include <mutex>
#include <iostream>
#include "base/exaudflib/swig/swig_common.h"
#include "base/exaudflib/swig/script_data_transfer_objects_wrapper.h"

namespace SWIGVMContainers {

class SWIGVM {
    public:
        struct exception : public std::exception {
            exception(const char *reason): m_reason(reason) { }
            exception(const std::string& reason): m_reason(reason) { }
            virtual ~exception() throw() { }
            const char* what() const throw() { return m_reason.c_str(); }
            private:
                std::string m_reason;
        };
        SWIGVM() { }
        virtual ~SWIGVM() { }
        virtual void shutdown() {};
        virtual void destroy() {};
        virtual bool run() = 0;
        bool run_() {
            try {
                return run();
            } catch (SWIGVM::exception& ex) {
                std::cerr << "SWGVM run_: caught: " << ex.what();
                std::lock_guard<std::mutex> lck(exception_msg_mtx);
                exception_msg = ex.what();
                return true; /* we are done */
            }
        }
        virtual bool useZmqSocketLocks() {return false;}
        virtual const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) = 0;
        std::string exception_msg;
        std::mutex exception_msg_mtx;
        std::string calledUndefinedSingleCall;
};

} //namespace SWIGVMContainers

#endif //SWIGVM_H