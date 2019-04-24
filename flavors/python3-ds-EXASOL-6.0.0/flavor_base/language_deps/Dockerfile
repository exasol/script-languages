FROM {{udfclient_deps}}

RUN mkdir -p /packages
COPY language_deps/packages /packages/language_deps

# ppa:deadsnakes/ppa needed for python3.6 under ubuntu:16.04
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install $(cat /packages/language_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN /scripts/install_packages.sh "PYTHON3-PIP3" "python3.6 -m pip install" /packages/language_deps/pip3_packages

ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.6

