FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=de.
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts
RUN mkdir -p /packages
COPY flavor_base_deps/packages /packages/flavor_base_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN /scripts/install_packages.sh "R" "Rscript -e" /packages/flavor_base_deps/r_packages
RUN /scripts/install_packages.sh "PYTHON2-PIP" "pip install" /packages/flavor_base_deps/pip_packages
RUN /scripts/install_packages.sh "PYTHON3-PIP3" "pip3 install" /packages/flavor_base_deps/pip3_packages