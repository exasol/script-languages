#ifndef EXAUDFCLIENT_EXAUDFLIB_CHECK_H
#define EXAUDFCLIENT_EXAUDFLIB_CHECK_H


namespace exaudflib_check {
    void init_socket_name(const char* the_socket_name);
    void init_socket_name_file(const char* the_socket_name_file);
    const char* get_socket_name_file();
    const char* get_socket_name_str();
    void external_process_check();
    void start_check_thread();
    void stop_check_thread();
    void cancel_check_thread();
    void set_remote_client(bool value);
    bool get_remote_client();
};


#endif //EXAUDFCLIENT_EXAUDFLIB_CHECK_H
