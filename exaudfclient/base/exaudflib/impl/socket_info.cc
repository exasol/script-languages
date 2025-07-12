#include "base/exaudflib/impl/socket_info.h"


namespace exaudflib {
    namespace socket_info {
        static const char *socket_file_name;
        static const char *socket_name_url;
    }
}


void exaudflib::socket_info::set_socket_url(const char* the_socket_url) {
    socket_name_url = the_socket_url;
}

const char* exaudflib::socket_info::get_socket_url() {
    return socket_name_url;
}

void exaudflib::socket_info::set_socket_file_name(const char* the_socket_file_name) {
    socket_file_name = the_socket_file_name;
}

const char* exaudflib::socket_info::get_socket_file_name() {
    return socket_file_name;
}
