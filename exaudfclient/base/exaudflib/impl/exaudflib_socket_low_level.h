#ifndef EXAUDFCLIENT_EXAUDFLIB_SOCKET_LOW_LEVEL_H
#define EXAUDFCLIENT_EXAUDFLIB_SOCKET_LOW_LEVEL_H

#include <zmq.hpp>

namespace exaudflib_socket_low_level {
    void init(bool use_zmq_socket_locks);
    void socket_send(zmq::socket_t &socket, zmq::message_t &zmsg);
    bool socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error=false);
}


#endif //EXAUDFCLIENT_EXAUDFLIB_SOCKET_LOW_LEVEL_H
