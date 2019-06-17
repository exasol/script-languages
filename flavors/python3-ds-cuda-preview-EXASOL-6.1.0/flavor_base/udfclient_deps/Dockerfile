FROM {{cuda_deps}}
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN mkdir -p /build_info/packages
COPY udfclient_deps/packages /build_info/packages/udfclient_deps

RUN apt-get update

RUN apt-get update && \
    apt-get -y install $(cat /build_info/packages/udfclient_deps/apt_get_packages) && \
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

ENV PROTOBUF_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu
ENV PROTOBUF_INCLUDE_PREFIX=/usr/include/google/protobuf/