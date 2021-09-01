#include "exaudflib_check.h"
#include <pthread.h>
#include <unistd.h>
#include <iostream>

#include "debug_message.h"

static const char *socket_name_file;
static const char *socket_name_str;
static bool keep_checking = true;
static pthread_t check_thread;
static int first_ppid=-1;
static bool remote_client;

void exaudflib_check::init_socket_name(const char* the_socket_name) {
    socket_name_str = the_socket_name;
}

const char* exaudflib_check::get_socket_name_str() {
    return socket_name_str;
}

void exaudflib_check::init_socket_name_file(const char* the_socket_name_file) {
    socket_name_file = the_socket_name_file;
}

const char* exaudflib_check::get_socket_name_file() {
    return socket_name_file;
}


void exaudflib_check::set_remote_client(bool value) {
    remote_client = value;
}

bool exaudflib_check::get_remote_client() {
    return remote_client;
}


void exaudflib_check::external_process_check() {
    if (remote_client) return;
    if (::access(socket_name_file, F_OK) != 0) {
        ::sleep(1); // give me a chance to die with my parent process
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1000","exaudfclient aborting ... cannot access socket file " << socket_name_str+6 << ".");
        DBG_STREAM_MSG(std::cerr,"### SWIGVM aborting with name '" << socket_name_str << "' (" << ::getppid() << ',' << ::getpid() << ')');
        ::abort();
    }
}


static void check_parent_pid(){
    int new_ppid=::getppid();
    if(first_ppid==-1){ // Initialize first_ppid
        first_ppid=new_ppid;
    }
    // Check if ppid has changed, if client is in own namespace,
    // the ppid will be forever 0 and never change.
    // If the client runs as udfplugin the ppid will point to the exasql process
    // and will change if it gets killed. Then client gets an orphaned process and
    // will be adopted by another process
    if(first_ppid!=new_ppid){
        ::sleep(1); // give me a chance to die with my parent process
        PRINT_ERROR_MESSAGE(std::cerr,"F-UDF-CL-LIB-1001","exaudfclient aborting " << socket_name_str << " ... current parent pid " << new_ppid << " different to first parent pid " << first_ppid << "." );
        DBG_STREAM_MSG(std::cerr,"### SWIGVM aborting with name '" << socket_name_str << "' (" << ::getppid() << ',' << ::getpid() << ')');
        ::unlink(socket_name_file);
        ::abort();
    }
}

static void *check_thread_routine(void* data)
{
    while(keep_checking) {
        exaudflib_check::external_process_check();
        check_parent_pid();
        ::usleep(100000);
    }
    return NULL;

}

void exaudflib_check::start_check_thread() {
    if (!remote_client)
        pthread_create(&check_thread, NULL, check_thread_routine, NULL);
}

void exaudflib_check::stop_check_thread() {
    keep_checking = false;
}

void exaudflib_check::cancel_check_thread() {
    ::pthread_cancel(check_thread);
}
