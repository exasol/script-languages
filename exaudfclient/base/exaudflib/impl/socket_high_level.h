#ifndef EXAUDFLIB_SOCKET_HIGH_LEVEL_H
#define EXAUDFLIB_SOCKET_HIGH_LEVEL_H

#include <string>
#include <zmq.hpp>

namespace exaudflib {
    namespace socket_high_level {

        bool send_init(zmq::socket_t &socket, const std::string client_name);
        void send_close(zmq::socket_t &socket, const std::string &exmsg);
        bool send_run(zmq::socket_t &socket);
        void send_undefined_call(zmq::socket_t &socket, const std::string& fn);
        bool send_done(zmq::socket_t &socket);
        void send_finished(zmq::socket_t &socket);
        bool send_return(zmq::socket_t &socket, const char* result);

    } //socket_high_level
} //exaudflib


#endif //EXAUDFLIB_SOCKET_HIGH_LEVEL_H
