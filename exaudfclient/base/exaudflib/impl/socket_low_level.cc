#include "exaudflib/impl/socket_low_level.h"
#include "exaudflib/impl/check.h"
#include "debug_message.h"
#include <mutex>
#include <iostream>
#include <unistd.h>

namespace exaudflib {
    namespace socket_low_level {
        static std::mutex zmq_socket_mutex;
        static bool use_zmq_socket_locks = false;
    }
}

void exaudflib::socket_low_level::init(bool _use_zmq_socket_locks) {
    use_zmq_socket_locks = _use_zmq_socket_locks;
}

void exaudflib::socket_low_level::socket_send(zmq::socket_t &socket, zmq::message_t &zmsg) {
    DBG_FUNC_BEGIN(std::cerr);
#ifdef LOG_COMMUNICATION
    stringstream sb;
    uint32_t len = zmsg.size();
    sb << "/tmp/zmqcomm_log_" << ::getpid() << "_send.data";
    int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
    if (fd >= 0) {
        if (::write(fd, &len, sizeof(uint32_t)) == -1 ) {perror("Log communication");}
        if (::write(fd, zmsg.data(), len) == -1) {perror("Log communication");}
        ::close(fd);
    }
#endif
    for (;;) {
        try {
            if (use_zmq_socket_locks) {
                zmq_socket_mutex.lock();
            }
            if (socket.send(zmsg) == true) {
                if (use_zmq_socket_locks) {
                    zmq_socket_mutex.unlock();
                }
                return;
            }
            exaudflib::check::external_process_check();
        } catch (std::exception &err) {
            exaudflib::check::external_process_check();
        } catch (...) {
            exaudflib::check::external_process_check();
        }
        if (use_zmq_socket_locks) {
            zmq_socket_mutex.unlock();
        }
        ::usleep(100000);
    }
    if (use_zmq_socket_locks) {
        zmq_socket_mutex.unlock();
    }
}

bool exaudflib::socket_low_level::socket_recv(zmq::socket_t &socket, zmq::message_t &zmsg, bool return_on_error)
{
    DBG_FUNC_BEGIN(std::cerr);
    for (;;) {
        try {
            if (use_zmq_socket_locks) {
                zmq_socket_mutex.lock();
            }
            if (socket.recv(&zmsg) == true) {
#ifdef LOG_COMMUNICATION
                stringstream sb;
                uint32_t len = zmsg.size();
                sb << "/tmp/zmqcomm_log_" << ::getpid() << "_recv.data";
                int fd = ::open(sb.str().c_str(), O_CREAT | O_APPEND | O_WRONLY, 00644);
                if (fd >= 0) {
                    if (::write(fd, &len, sizeof(uint32_t)) == -1) {perror("Log communication");}
                    if (::write(fd, zmsg.data(), len) == -1) {perror("Log communication");}
                    ::close(fd);
                }
#endif
                if (use_zmq_socket_locks) {
                    zmq_socket_mutex.unlock();
                }
                return true;
            }
            exaudflib::check::external_process_check();
        } catch (std::exception &err) {
            exaudflib::check::external_process_check();

        } catch (...) {
            exaudflib::check::external_process_check();
        }
        if (use_zmq_socket_locks) {
            zmq_socket_mutex.unlock();
        }
        if (return_on_error) return false;
        ::usleep(100000);
    }
    if (use_zmq_socket_locks) {
        zmq_socket_mutex.unlock();
    }
    return false;
}