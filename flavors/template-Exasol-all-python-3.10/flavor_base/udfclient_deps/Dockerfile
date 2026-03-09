FROM exasol/script-language-container:exaslpm-1.0.0-ubuntu-22.04
ENV DEBIAN_FRONTEND=noninteractive

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY udfclient_deps_packages.yaml /build_info/packages/udfclient_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/udfclient_deps_packages.yaml --build-step udfclient_deps

RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution --gecos "First Last,RoomNumber,WorkPhone,HomePhone"
RUN addgroup --gid 500 exausers
RUN adduser --disabled-login --uid 500 --gid 500 exadefusr --gecos "First Last,RoomNumber,WorkPhone,HomePhone"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
