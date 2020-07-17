FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps
COPY language_deps/packages/apt_get_packages /build_info/packages/language_deps

# ppa:deadsnakes/ppa needed for python3.6 under ubuntu:16.04
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install $(cat /build_info/packages/language_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py
COPY scripts /scripts
COPY language_deps/packages/pip3_packages /build_info/packages/language_deps
RUN /scripts/install_scripts/install_packages_batch.sh "PYTHON3-PIP" "pip install --ignore-installed <<list>>" "<<package>>" " " /build_info/packages/language_deps/pip3_packages

ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.6

