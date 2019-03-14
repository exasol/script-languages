FROM {{build_deps}}

RUN apt-get -y update && \
    apt-get -y install python-protobuf python-zmq gdb valgrind gdbserver binutils patchelf chrpath strace && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig