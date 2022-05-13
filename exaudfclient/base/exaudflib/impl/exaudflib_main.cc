#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <iostream>
#include <sstream>
#include <sstream>
#include <zmq.hpp>
#include <fcntl.h>
#include <fstream>
#include <functional>

#include "debug_message.h"

// swig lib
#include <limits>


#include "exaudflib/impl/check.h"
#include "exaudflib/impl/socket_info.h"
#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/impl/msg_conversion.h"
#include "exaudflib/impl/global.h"
#include "exaudflib/impl/socket_high_level.h"
#include "exaudflib/impl/swig/swig_parameter.h"

#include "exaudflib/vm/swig_vm.h"


#ifndef NDEBUG
#define SWIGVM_LOG_CLIENT
#endif
//#define SWIGVM_LOG_CLIENT
//#define LOG_COMMUNICATION


void print_args(int argc,char**argv){
    for (int i = 0; i<argc; i++)
    {
        std::cerr << "zmqcontainerclient argv[" << i << "] = " << argv[i] << std::endl;
    }
}


void delete_vm(SWIGVMContainers::SWIGVM*& vm){
    if (vm != nullptr)
    {
        delete vm;
        vm = nullptr;
    }
}

void stop_all(zmq::socket_t& socket){
    socket.close();
    exaudflib::check::stop_check_thread();
    if (!exaudflib::check::get_remote_client()) {
        exaudflib::check::cancel_check_thread();
        ::unlink(exaudflib::socket_info::get_socket_file_name());
    } else {
        ::sleep(3); // give other components time to shutdown
    }
}

unsigned int handle_error(zmq::socket_t& socket, std::string socket_name, SWIGVMContainers::SWIGVM* vm, std::string msg, bool shutdown_vm=false){
    DBG_STREAM_MSG(std::cerr,"### handle error in '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << msg);
    try{
        if(vm!=nullptr && shutdown_vm){
            vm->exception_msg = "";
            vm->shutdown(); // Calls cleanup
            if (vm->exception_msg.size()>0) {
                PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1110","### Caught error in vm->shutdown '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << vm->exception_msg);
                msg ="F-UDF-CL-LIB-1111: Caught exception\n\n"+msg+"\n\n and caught another exception during cleanup\n\n"+vm->exception_msg;
            }
        } 
        delete_vm(vm);
    }  catch (SWIGVMContainers::SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1112","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1113","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1114","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): ");
    }
    try{
        exaudflib::socket_high_level::send_close(socket, msg);
        ::sleep(1); // give me a chance to die with my parent process
    }  catch (SWIGVMContainers::SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1115","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1116","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1117","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ")");
    }

    try{
        stop_all(socket);
    }  catch (SWIGVMContainers::SWIGVM::exception &err) {
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1118","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(std::exception& err){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1119","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
    }catch(...){
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1120","### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "):");
    }
    return 1;
}

extern "C" {

int exaudfclient_main(std::function<SWIGVMContainers::SWIGVM*()>vmMaker,int argc,char**argv)
{
    assert(exaudflib::global.SWIGVM_params_ref != nullptr);

#ifdef UDF_PLUGIN_CLIENT
    std::stringstream socket_name_ss;
#endif
    std::string socket_name = argv[1];
    exaudflib::socket_info::set_socket_file_name(argv[1]);
    exaudflib::socket_info::set_socket_url(argv[1]);

    exaudflib::check::set_remote_client(false);

    zmq::context_t context(1);

    DBG_COND_FUNC_CALL(std::cerr, print_args(argc,argv));

    if (socket_name.length() > 4 ) {
#ifdef UDF_PLUGIN_CLIENT
        // udf plugins might not have arguments
#else
        if (! ((strcmp(argv[2], "lang=python") == 0)
               || (strcmp(argv[2], "lang=r") == 0)
               || (strcmp(argv[2], "lang=java") == 0)
               || (strcmp(argv[2], "lang=streaming") == 0)
               || (strcmp(argv[2], "lang=benchmark") == 0)) )
        {
            PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1121","Remote VM type '" << argv[2] << "' not supported.");
            return 2;
        }
#endif
    } else {
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1122", "socket name '" << socket_name << "' is invalid." );
        abort();
    }

    if (socket_name.compare(0, 4, "tcp:") == 0) {
        exaudflib::check::set_remote_client(true);
    }

    if (socket_name.compare(0, 4, "ipc:") == 0)
    {        
#ifdef UDF_PLUGIN_CLIENT
/*
    DO NOT REMOVE, required for Exasol 6.2
*/
        //In case of protegrity the /tmp folder can't be used as it runs within the database (and not in an isolated container).
        //There, the actual socket file path is defined in a environment variable. We need to replace it here.
        if (socket_name.compare(0, 11, "ipc:///tmp/") == 0) {
            socket_name_ss << "ipc://" << getenv("NSEXEC_TMP_PATH") << '/' << &(socket_name.c_str()[11]);
            socket_name = socket_name_ss.str();
            socket_info::set_socket_file_name(::strdup(socket_name.c_str()));
        }
#endif
        exaudflib::socket_info::set_socket_file_name(&(exaudflib::socket_info::get_socket_file_name()[6]));
    }

    DBG_STREAM_MSG(std::cerr,"### SWIGVM starting " << argv[0] << " with name '" << socket_name <<
                   " (" << ::getppid() << ',' << ::getpid() << "): '" << argv[1] << '\'');

    exaudflib::check::start_check_thread();


    int linger_timeout = 0;
    int recv_sock_timeout = 1000;
    int send_sock_timeout = 1000;

    if (exaudflib::check::get_remote_client()) {
        recv_sock_timeout = 10000;
        send_sock_timeout = 5000;
    }

reinit:

    DBGMSG(std::cerr,"Reinit");
    zmq::socket_t socket(context, ZMQ_REQ);

    socket.setsockopt(ZMQ_LINGER, &linger_timeout, sizeof(linger_timeout));
    socket.setsockopt(ZMQ_RCVTIMEO, &recv_sock_timeout, sizeof(recv_sock_timeout));
    socket.setsockopt(ZMQ_SNDTIMEO, &send_sock_timeout, sizeof(send_sock_timeout));

    if (exaudflib::check::get_remote_client()) socket.bind(socket_name.c_str());
    else socket.connect(socket_name.c_str());

    exaudflib::global.SWIGVM_params_ref->sock = &socket;
    exaudflib::global.SWIGVM_params_ref->exch = &exaudflib::global.exchandler;

    SWIGVMContainers::SWIGVM* vm=nullptr;

    if (!exaudflib::socket_high_level::send_init(socket, socket_name)) {
        if (!exaudflib::check::get_remote_client() && exaudflib::global.exchandler.exthrowed) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1123: " +
                                exaudflib::global.exchandler.exmsg, false);
        }else{
            goto reinit;
        }
    }

    exaudflib::global.initSwigParams();

    bool shutdown_vm_in_case_of_error = false;
    try {
        vm = vmMaker();
        if (vm == nullptr) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1124: Unknown or unsupported VM type", false);
        }
        if (vm->exception_msg.size()>0) {
            return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1125: "+vm->exception_msg, false);
        }
        shutdown_vm_in_case_of_error = true;
        exaudflib::socket_low_level::init(vm->useZmqSocketLocks());
        if (exaudflib::global.singleCallMode) {
            ExecutionGraph::EmptyDTO noArg; // used as dummy arg
            for (;;) {
                // in single call mode, after MT_RUN from the client,
                // EXASolution responds with a CALL message that specifies
                // the single call function to be made
                if (!exaudflib::socket_high_level::send_run(socket)) {
                    break;
                }

                assert(exaudflib::global.singleCallFunction != SWIGVMContainers::single_call_function_id_e::SC_FN_NIL);
                try {
                    const char* result = nullptr;
                    switch (exaudflib::global.singleCallFunction)
                    {
                        case SWIGVMContainers::single_call_function_id_e::SC_FN_NIL:
                            break;
                        case SWIGVMContainers::single_call_function_id_e::SC_FN_DEFAULT_OUTPUT_COLUMNS:
                            result = vm->singleCall(exaudflib::global.singleCallFunction,noArg);
                            break;
                        case SWIGVMContainers::single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
                            assert(!exaudflib::global.singleCall_ImportSpecificationArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_ImportSpecificationArg);
                            exaudflib::global.singleCall_ImportSpecificationArg = ExecutionGraph::ImportSpecification();  // delete the last argument
                            break;
                        case SWIGVMContainers::single_call_function_id_e::SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC:
                            assert(!exaudflib::global.singleCall_ExportSpecificationArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_ExportSpecificationArg);
                            exaudflib::global.singleCall_ExportSpecificationArg = ExecutionGraph::ExportSpecification();  // delete the last argument
                            break;
                        case SWIGVMContainers::single_call_function_id_e::SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
                            assert(!exaudflib::global.singleCall_StringArg.isEmpty());
                            result = vm->singleCall(exaudflib::global.singleCallFunction,exaudflib::global.singleCall_StringArg);
                            break;
                    }
                    if (vm->exception_msg.size()>0) {
                        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1126: "+vm->exception_msg,true);
                    }

                    if (vm->calledUndefinedSingleCall.size()>0) {
                        exaudflib::socket_high_level::send_undefined_call(socket, vm->calledUndefinedSingleCall);
                    } else {
                        exaudflib::socket_high_level::send_return(socket,result);
                    }

                    if (!exaudflib::socket_high_level::send_done(socket)) {
                        break;
                    }
                } catch(...) {}
            }
        } else {
            for(;;) {
                if (!exaudflib::socket_high_level::send_run(socket))
                    break;
                exaudflib::global.SWIGVM_params_ref->inp_force_finish = false;
                while(!vm->run_())
                {
                    if (vm->exception_msg.size()>0) {
                        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1127: "+vm->exception_msg,true);
                    }
                }
                if (!exaudflib::socket_high_level::send_done(socket))
                    break;
            }
        }

        if (vm != nullptr)
        {
            vm->shutdown();
            if (vm->exception_msg.size()>0) {
                return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1128: "+vm->exception_msg,false);
            }
        }
        exaudflib::socket_high_level::send_finished(socket);
    }  catch (SWIGVMContainers::SWIGVM::exception &err) {
        DBG_STREAM_MSG(std::cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1129: "+std::string(err.what()),shutdown_vm_in_case_of_error);
    } catch (std::exception &err) {
        DBG_STREAM_MSG(std::cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << "): " << err.what());
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1130: "+std::string(err.what()),shutdown_vm_in_case_of_error);
    } catch (...) {
        DBG_STREAM_MSG(std::cerr,"### SWIGVM crashing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ')');
        return handle_error(socket, socket_name, vm, "F-UDF-CL-LIB-1131: Internal/Unknown error",shutdown_vm_in_case_of_error);
    }

    DBG_STREAM_MSG(std::cerr,"### SWIGVM finishing with name '" << socket_name << " (" << ::getppid() << ',' << ::getpid() << ')');

    delete_vm(vm);
    stop_all(socket);
    return 0;
}


} // extern "C"
