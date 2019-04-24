FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=de.
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts
RUN mkdir -p /packages
COPY flavor_base_deps/packages /packages/flavor_base_deps

# ppa:deadsnakes/ppa needed for python3.6 under ubuntu:16.0
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install $(cat /packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py
RUN /scripts/install_packages.sh "PYTHON3-PIP3" "pip install --ignore-installed " /packages/flavor_base_deps/pip3_packages