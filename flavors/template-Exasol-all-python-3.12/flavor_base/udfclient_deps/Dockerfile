FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY udfclient_deps/packages /build_info/packages/udfclient_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/udfclient_deps/apt_get_packages --with-versions

RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution --gecos "First Last,RoomNumber,WorkPhone,HomePhone"
RUN addgroup --gid 500 exausers
RUN adduser --disabled-login --uid 500 --gid 500 exadefusr --gecos "First Last,RoomNumber,WorkPhone,HomePhone"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PROTOBUF_LIBRARY_PREFIX=/usr/lib/
ENV PROTOBUF_INCLUDE_PREFIX=/usr/include/
ENV ZMQ_LIBRARY_PREFIX=/usr/lib
ENV ZMQ_INCLUDE_PREFIX=/usr/include
