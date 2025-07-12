#include "base/exaudflib/impl/check.h"
#include "base/exaudflib/impl/socket_info.h"
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <atomic>


#include "base/utils/debug_message.h"

namespace exaudflib {
    namespace check {
        static std::atomic_bool keep_checking(true);
        static pthread_t check_thread;
        static int first_ppid=-1;
        static bool remote_client;
    }
}

void exaudflib::check::set_remote_client(bool value) {
    remote_client = value;
}

bool exaudflib::check::get_remote_client() {
    return remote_client;
}

void exaudflib::check::external_process_check() {
    if (remote_client) return;
    if (::access(socket_info::get_socket_file_name(), F_OK) != 0) {
        ::sleep(1); // give me a chance to die with my parent process
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1000","exaudfclient aborting ... cannot access socket file " << socket_info::get_socket_url()+6 << ".");
        DBG_STREAM_MSG(std::cerr,"### SWIGVM aborting with name '" << socket_info::get_socket_url() << "' (" << ::getppid() << ',' << ::getpid() << ')');
        ::abort();
    }
}


static void check_parent_pid(){
    int new_ppid=::getppid();
    if(exaudflib::check::first_ppid==-1){ // Initialize first_ppid
        exaudflib::check::first_ppid=new_ppid;
    }
    // Check if ppid has changed, if client is in own namespace,
    // the ppid will be forever 0 and never change.
    // If the client runs as udfplugin the ppid will point to the exasql process
    // and will change if it gets killed. Then client gets an orphaned process and
    // will be adopted by another process
    if(exaudflib::check::first_ppid!=new_ppid){
        ::sleep(1); // give me a chance to die with my parent process
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1001","exaudfclient aborting " << exaudflib::socket_info::get_socket_url() << " ... current parent pid " << new_ppid << " different to first parent pid " << exaudflib::check::first_ppid << "." );
        DBG_STREAM_MSG(std::cerr,"### SWIGVM aborting with name '" << exaudflib::socket_info::get_socket_url() << "' (" << ::getppid() << ',' << ::getpid() << ')');
        ::unlink(exaudflib::socket_info::get_socket_file_name());
        ::abort();
    }
}

static void *check_thread_routine(void* data)
{
    while(exaudflib::check::keep_checking) {
        exaudflib::check::external_process_check();
        check_parent_pid();
        ::usleep(100000);
    }
    return NULL;
}

void exaudflib::check::start_check_thread() {
    if (!remote_client)
        pthread_create(&check_thread, NULL, check_thread_routine, NULL);
}

void exaudflib::check::stop_check_thread() {
    keep_checking = false;
}

void exaudflib::check::cancel_check_thread() {
    ::pthread_cancel(check_thread);
}
