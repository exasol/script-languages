FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY nvidia_driver_deps/packages /build_info/packages/nvidia_driver_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/nvidia_driver_deps/apt_get_packages --with-versions

#ENV NVIDIA_DRIVER_VERSION=550.54.15
#RUN /scripts/install_scripts/install_nvidia_driver_libs.sh "$NVIDIA_DRIVER_VERSION"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


