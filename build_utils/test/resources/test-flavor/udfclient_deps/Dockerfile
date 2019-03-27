FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive

COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN apt-get -y update && \
    apt-get -y install \
        coreutils \
        locales \
        libzmq3-dev \
        protobuf-compiler \
        libprotobuf-dev \
        libssl-dev \
        unzip \
        libpcre3-dev \
        curl \
        libnss-db && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution --gecos "First Last,RoomNumber,WorkPhone,HomePhone"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PROTOBUF_BIN=/usr/bin/protoc
ENV PROTOBUF_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu
ENV PROTOBUF_INCLUDE_PREFIX=/usr/include/google/protobuf/