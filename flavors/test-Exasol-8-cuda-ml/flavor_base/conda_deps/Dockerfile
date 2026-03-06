FROM exasol/script-language-container:exaslpm-ubuntu-24.04
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY conda_deps_packages.yaml /build_info/packages/conda_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/conda_deps_packages.yaml --build-step conda_deps



RUN usermod -l exasolution ubuntu
RUN groupmod -n exasolution ubuntu
RUN addgroup --gid 500 exausers
RUN adduser --disabled-login --uid 500 --gid 500 exadefusr --gecos "First Last,RoomNumber,WorkPhone,HomePhone"


ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
