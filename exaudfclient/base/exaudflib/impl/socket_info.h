#ifndef EXAUDFCLIENT_SOCKET_INFO_H
#define EXAUDFCLIENT_SOCKET_INFO_H

namespace exaudflib {
    namespace socket_info {
        void set_socket_url(const char*);
        void set_socket_file_name(const char*);
        const char* get_socket_file_name();
        const char* get_socket_url();
    }
}


#endif //EXAUDFCLIENT_SOCKET_INFO_H
